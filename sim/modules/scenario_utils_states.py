import json
import random
from copy import deepcopy

from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer

from agents.agent import Agent
from sim.modules.graph_utils import SimilarityGraph
from sim.modules.utils import read_prompts, save_to_disk, run_agent_query, check_for_missing_fields, load_output_schemas
from sim.modules.utils import json_obj_list_to_dict


def get_valid_scenarios(roles: dict, required_fields: list, min_scenarios_per_role: int = None):
    failed_roles = check_for_missing_fields(roles, required_fields)
    if failed_roles:
        roles = {k: v for k, v in roles.items() if v['name'] not in failed_roles}
    if min_scenarios_per_role:
        roles = {k: v for k, v in roles.items() if len(v['scenarios']) >= min_scenarios_per_role}
    return roles


def is_judgment_valid(judged_role: dict, scores_fields: list, scores_range: tuple):
    try:
        assert 'name' in judged_role
        assert len(judged_role['aligned_scenarios']) == 3
        assert len(judged_role['misaligned_scenarios']) == 3

        for scenario_type in ['aligned_scenarios', 'misaligned_scenarios']:
            for scenario in judged_role[scenario_type]:
                assert 'scenario_name' in scenario

                for score_field in scores_fields:
                    assert score_field in scenario
                    assert scores_range[0] <= scenario[score_field] <= scores_range[1]
        return True
    except Exception as e:
        print(f"Error in is_judgment_valid: {e}")
    return False


def normalize_scenarios(scenarios: dict) -> dict:
    # TODO
    # for role in scenarios.values():
    #     for scenario_type in ['aligned_scenarios', 'misaligned_scenarios']:
    #         for scenario in role[scenario_type]:
    #             scenario['scenario_name'] = normalize_string(scenario['scenario_name'], lower=False)
    #             scenario['scenario_detailed_description'] = normalize_string(
    #                 scenario['scenario_detailed_description'])
    #             scenario['scenario_outcome'] = normalize_string(scenario['scenario_outcome'])

    return scenarios


def find_acceptable_scenarios(judged_scenarios: dict):
    out = {}
    for role_name, role_data in judged_scenarios.items():
        aligned_scenarios, misaligned_scenarios = [], []
        for scenario_type, scenario_type_array in zip(['aligned_scenarios', 'misaligned_scenarios'],
                                                      [aligned_scenarios, misaligned_scenarios]):
            for scenario in role_data[scenario_type]:
                if all(scenario[score] >= 6 for score in
                       ['clarity_score', 'alignment_score', 'simplicity_score', 'realism_score']):
                    scenario_type_array.append(scenario)
        if aligned_scenarios or misaligned_scenarios:
            out[role_name] = role_data
            out[role_name]['aligned_scenarios'] = aligned_scenarios
            out[role_name]['misaligned_scenarios'] = misaligned_scenarios

    return out


class ScenarioManager:
    def __init__(self, api_conf, logger, workspace_name: str, workspace_desc: str, workspace_alternative_forms: list,
                 domain_name, domain_desc: str,
                 domain_alternative_forms: list, prompts_conf: DictConfig,
                 output_schemas_conf: DictConfig, object_storage_conf: DictConfig, temperature,
                 min_initial_scenarios_per_role: int, min_chosen_scenarios_per_role: int, generation_batch_size: int):
        self.api_conf = api_conf
        self.logger = logger
        self.workspace = workspace_name
        self.workspace_desc = workspace_desc
        self.workspace_alternative_forms = workspace_alternative_forms
        self.domain = domain_name
        self.domain_desc = domain_desc
        self.domain_alternative_forms = domain_alternative_forms
        self.prompts_conf = prompts_conf
        self.output_schemas_conf = output_schemas_conf
        self.object_storage_conf = object_storage_conf
        self.temperature = temperature

        self.min_initial_scenarios_per_role = min_initial_scenarios_per_role
        self.min_chosen_scenarios_per_role = min_chosen_scenarios_per_role
        self.batch_size = generation_batch_size

        self.scenarios_generation_agent = self._init_scenarios_generation_agent()
        self.scenarios_verif_judge = self._init_scenarios_verif_judge()

        self.similarity_graph = SimilarityGraph(TfidfVectorizer, threshold=0.5)

    def _init_scenarios_generation_agent(self):
        sys_prompt = read_prompts(self.prompts_conf.scenarios_agents_states, key='SYS_GEN',
                                  context={'workspace': self.workspace, 'workspace_desc': self.workspace_desc,
                                           'domain': self.domain, 'domain_desc': self.domain_desc,
                                           'n_scenarios': self.min_initial_scenarios_per_role}, logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_states)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def _init_scenarios_verif_judge(self):
        sys_prompt = read_prompts(self.prompts_conf.judge_agents, key='SYS_SCENARIOS_STATES_VERIF',
                                  context={'workspace': self.workspace, 'workspace_desc': self.workspace_desc,
                                           'domain': self.domain, 'domain_desc': self.domain_desc},
                                  logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.judge_scenarios)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def remove_similar_scenarios(self, roles_with_scenarios: dict, min_chosen_scenarios_per_role: int):
        out_roles = deepcopy(roles_with_scenarios)
        for role_k, role_v in roles_with_scenarios.items():
            curr_scenarios = role_v['scenarios']
            # Form the similarity graph and remove similar entries
            graph_data = {scenario['name']: scenario['description'] for scenario in curr_scenarios}
            similarity_graph = SimilarityGraph(TfidfVectorizer, threshold=0.5)
            G = similarity_graph.detect_and_remove_similar_entries(graph_data)
            out_roles[role_k]['scenarios'] = [x for x in curr_scenarios if x['name'] in G.nodes()]

        out_roles = {k: v for k, v in out_roles.items() if len(v['scenarios']) >= min_chosen_scenarios_per_role}
        return out_roles

    def generate_scenarios(self, input_roles):
        roles_with_scenarios = deepcopy(input_roles)
        roles_to_process = list(input_roles.keys())
        batch_size = self.batch_size

        while roles_to_process:
            if self.logger:
                self.logger.debug(f"Roles left to process: {roles_to_process}")

            batch_roles = roles_to_process[:batch_size]
            random.shuffle(batch_roles)
            prompt = read_prompts(self.prompts_conf.scenarios_agents_states, key='USER_GEN',
                                  context={'roles': {role: input_roles[role] for role in batch_roles}},
                                  logger=self.logger)
            try:
                response = run_agent_query(prompt=prompt, agent=self.scenarios_generation_agent,
                                           logger=self.logger,
                                           to_json=True, json_transform_keys=['roles', 'name'])
            except json.decoder.JSONDecodeError as e:
                self.logger.error(f"JSONDecodeError in generate_scenarios: {type(e)}:{e}")
                batch_size = max(1, batch_size // 2)
                continue
            except Exception as e:
                self.logger.error(f"Error in generate_scenarios: {e}")
                continue

            try:
                # response = get_valid_scenarios(response, required_fields=['name', 'scenarios:role_policy',
                #                                                           'scenarios:initial_state', 'scenarios:illegal_state',
                #                                                           'scenarios:trigger_action'])
                response = get_valid_scenarios(response, required_fields=['name', 'scenarios'],
                                               min_scenarios_per_role=self.min_initial_scenarios_per_role)
                response = self.remove_similar_scenarios(response,
                                                         min_chosen_scenarios_per_role=self.min_chosen_scenarios_per_role)
                for role in response:
                    response[role]['scenarios'] = json_obj_list_to_dict(response[role]['scenarios'], 'name')
                if self.logger:
                    self.logger.debug(f"Valid generated scenarios: {response}")

                for role in response.values():
                    if role['name'] in input_roles:
                        roles_with_scenarios[role['name']].update(role)
                        roles_to_process.remove(role['name'])
            except Exception as e:
                self.logger.error(f"Error in generate_scenarios: {e}")

        return roles_with_scenarios

    def judge_scenarios(self, input_scenarios: dict):
        # Roles for which the scenarios have been given
        roles_to_process = list(input_scenarios.keys())
        out = deepcopy(input_scenarios)
        batch_size = self.batch_size

        while roles_to_process:
            if roles_to_process:
                self.logger.debug(f"Roles to process (judge): {roles_to_process}")
            batch_roles = roles_to_process[:batch_size]
            random.shuffle(batch_roles)
            prompt = read_prompts(self.prompts_conf.judge_agents, key='USER_SCENARIOS_VERIF',
                                  context={'roles': str({name: input_scenarios[name] for name in batch_roles})},
                                  logger=self.logger)

            try:
                response = run_agent_query(prompt=prompt, agent=self.scenarios_verif_judge, logger=self.logger,
                                           to_json=True,
                                           json_transform_keys=['roles', 'name'])
            except json.decoder.JSONDecodeError as e:
                self.logger.error(f"JSONDecodeError in judge_scenarios run_agent_query: {type(e)}:{e}")
                batch_size = max(1, batch_size // 2)
                continue
            except Exception as e:
                self.logger.error(f"Error in judge_scenarios run_agent_query: {type(e)}:{e}")
                continue

            if self.logger:
                self.logger.debug(f"Response from scenarios judge: {response}")

            try:
                response = get_valid_scenarios(response, required_fields=['name', 'scenarios'])
                response = {k: v for k, v in response.items() if set(x['name'] for x in v['scenarios']) ==
                            set(input_scenarios[k]['scenarios'].keys())}
                for role in response:
                    response[role]['scenarios'] = json_obj_list_to_dict(response[role]['scenarios'], 'name')

                for role in response.values():
                    if role['name'] in input_scenarios:
                        out[role['name']].update(role)
                        roles_to_process.remove(role['name'])
            except Exception as e:
                self.logger.error(f"Error in judge_roles: {e}")

            # roles_to_process = new_missing_roles

        return out

    def generate_and_judge_scenarios(self, input_roles: dict, logging=True):
        accepted_scenarios = {}
        missing_scenarios = list(set(input_roles.keys()))
        n_tries_for_role = 0
        while missing_scenarios:
            if n_tries_for_role >= 10:
                break
            n_tries_for_role += 1
            generated_scenarios = self.generate_scenarios({name: input_roles[name] for name in missing_scenarios})
            if logging:
                self.logger.debug(f'Generated scenarios_dict: {generated_scenarios}\n\n')

            judged_scenarios = self.judge_scenarios(generated_scenarios)

            for role_k, role_v in judged_scenarios.items():
                all_scenarios_accepted = all(scen['acceptable'] == True for scen in role_v['scenarios'].values())
                if all_scenarios_accepted:
                    accepted_scenarios[role_k] = generated_scenarios[role_k]
                    # Update accepted scenarios with judgment fields provided by the judge
                    for scenario_k, scenario_v in role_v['scenarios'].items():
                        accepted_scenarios[role_k]['scenarios'][scenario_k].update(scenario_v)
                    missing_scenarios.remove(role_k)
                else:
                    if logging:
                        self.logger.debug(f"Judgment not valid for role {role_k}: {role_v}\n\n")
            # curr_accepted_scenarios = generated_scenarios
            # Update accepted scenarios and missing scenarios
            # for role_k, role_v in curr_accepted_scenarios.items():
            #     if role_k not in accepted_scenarios:
            #         accepted_scenarios[role_k] = input_roles[role_k]
            #         accepted_scenarios[role_k]['scenarios'] = {}
            #
            #     for scenario_type in ['aligned_scenarios', 'misaligned_scenarios']:
            #         for scenario in role_v[scenario_type]:
            #             if scenario not in accepted_scenarios[role_k][scenario_type]:
            #                 if scenario not in accepted_scenarios[role_k][scenario_type]:
            #                     scenario.update(generated_scenarios[role_k][scenario_type][0])
            #                     accepted_scenarios[role_k][scenario_type].append(scenario)
            # accepted_scenarios[role_k]['aligned_scenarios'].extend(role_v['aligned_scenarios'])
            # accepted_scenarios[role_k]['misaligned_scenarios'].extend(role_v['misaligned_scenarios'])
            #
            # if len(accepted_scenarios[role_k]['aligned_scenarios']) >= 3 and \
            #         len(accepted_scenarios[role_k]['misaligned_scenarios']) >= 3:
            #     missing_scenarios.remove(role_k)
            # accepted_scenarios.update(curr_accepted_scenarios)
            # missing_scenarios = list(set(missing_scenarios) - set(accepted_scenarios.keys()))

            if logging:
                self.logger.debug(f'Accepted scenario names: {list(accepted_scenarios.keys())}\n\n')

        # accepted_scenarios = normalize_scenarios(accepted_scenarios)
        return accepted_scenarios
