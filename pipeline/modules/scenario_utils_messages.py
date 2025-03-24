import json
import random
from copy import deepcopy
import traceback
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer

from agents.agent import Agent
from .graph_utils import SimilarityGraph
from .utils import read_prompts, save_to_disk, run_agent_query, check_for_missing_fields, load_output_schemas
from .utils import json_obj_list_to_dict


def get_valid_scenarios(roles: dict, required_fields: list):
    failed_roles = check_for_missing_fields(roles, required_fields)
    if failed_roles:
        roles = {k: v for k, v in roles.items() if v['name'] not in failed_roles}
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
    #             scenario['scenario_description'] = normalize_string(
    #                 scenario['scenario_description'])
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
                 generation_batch_size: int):
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

        self.batch_size = generation_batch_size

        self.scenarios_generation_agent = self._init_scenarios_generation_agent()
        self.scenarios_verif_judge = self._init_scenarios_verif_judge()

        self.similarity_graph = SimilarityGraph(TfidfVectorizer, threshold=0.5)

    def _init_scenarios_generation_agent(self):
        general_body = read_prompts(self.prompts_conf.scenarios_general_body, key='SYS_GEN',
                                    context={'workspace': self.workspace, 'workspace_desc': self.workspace_desc,
                                             'domain': self.domain, 'domain_desc': self.domain_desc},
                                    logger=self.logger)

        sys_prompt = read_prompts(self.prompts_conf.scenarios_agents_messages, key='SYS_GEN',
                                  context={'general_body': general_body}, logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_messages)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def _init_scenarios_verif_judge(self):
        sys_prompt = read_prompts(self.prompts_conf.judge_agents, key='SYS_SCENARIOS_MESSAGES_VERIF',
                                  context={'workspace': self.workspace, 'workspace_desc': self.workspace_desc,
                                           'domain': self.domain, 'domain_desc': self.domain_desc},
                                  logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.judge_scenarios)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def check_configurations_are_valid(self, roles_with_scenarios: dict):
        valid_scenarios = {}
        for role_k, role_v in roles_with_scenarios.items():
            for scenario_k, scenario_v in role_v['scenarios'].items():
                try:
                    conf_known_dict = json.loads(scenario_v['configurations']['configuration_known'])
                    conf_unknown_dict = json.loads(scenario_v['configurations']['configuration_unknown'])

                    conf_known_dict = {k: v for k, v in conf_known_dict.items() if
                                       v and k not in conf_unknown_dict.keys()}
                    conf_unknown_dict = {k: v for k, v in conf_unknown_dict.items() if
                                         v and k not in conf_known_dict.keys()}

                    assert len(conf_known_dict) > 0
                    assert len(conf_unknown_dict) > 0

                    valid_scenarios[role_k] = role_v
                    valid_scenarios[role_k]['scenarios'][scenario_k]['configurations'][
                        'configuration_known'] = str(conf_known_dict).replace("'", '"')
                    # print(str(conf_known_dict).replace("'", '"'))
                    valid_scenarios[role_k]['scenarios'][scenario_k]['configurations'][
                        'configuration_unknown'] = str(conf_unknown_dict).replace("'", '"')
                    # print(str(conf_unknown_dict).replace("'", '"'))
                except Exception as e:
                    self.logger.error(
                        f"Error in check_configurations_are_valid for scenario {role_k}:{scenario_k}: {e}")
                    self.logger.error(traceback.format_exc())
                    continue
        return valid_scenarios

    # def check_messages_are_valid(self, roles_with_scenarios: dict):
    #     valid_scenarios = {}
    #     for role_k, role_v in roles_with_scenarios.items():
    #         for scenario_k, scenario_v in role_v['scenarios'].items():
    #             # try: ...
    #             pass
    #     return valid_scenarios

    def generate_scenarios(self, input_roles: dict):
        # print(f"generate_scenarios Input roles: {input_roles}")
        roles_with_scenarios = deepcopy(input_roles)
        curr_input_roles = deepcopy(input_roles)
        roles_to_process = list(curr_input_roles.keys())
        batch_size = self.batch_size

        while roles_to_process:
            if self.logger:
                self.logger.debug(f"Roles left to process: {roles_to_process}")

            batch_roles = roles_to_process[:batch_size]
            random.shuffle(batch_roles)
            prompt = read_prompts(self.prompts_conf.scenarios_agents_messages, key='USER_GEN',
                                  context={'roles': {role: curr_input_roles[role] for role in batch_roles}},
                                  logger=self.logger)
            try:
                response = run_agent_query(prompt=prompt, agent=self.scenarios_generation_agent,
                                           logger=self.logger,
                                           to_json=True, json_transform_keys=['roles', 'name'])
            except Exception as e:
                self.logger.error(f"Error in generate_scenarios: {e}")
                self.logger.error(traceback.format_exc())
                batch_size = max(1, batch_size // 2)
                continue

            try:
                # Run validity checks
                response = get_valid_scenarios(response, required_fields=['name', 'scenarios'])
                response = {k: v for k, v in response.items() if set(x['name'] for x in v['scenarios']) ==
                            set(curr_input_roles[k]['scenarios'].keys())}
                response = {k: v for k, v in response.items() if
                            all(10 <= len(scen_v['sys_messages']) <= 18 for scen_v in v['scenarios'])}
                response = {k: v for k, v in response.items() if
                            all(len(scen_v['neutral_sys_messages']) >= 2 for scen_v in v['scenarios'])}

                for role in response:
                    response[role]['scenarios'] = json_obj_list_to_dict(response[role]['scenarios'], 'name')

                if self.logger:
                    self.logger.debug(f"Valid generated scenarios: {response}")

                for role in response.values():
                    if set(role['scenarios'].keys()) == set(curr_input_roles[role['name']]['scenarios'].keys()):
                        roles_to_process.remove(role['name'])
                        # Preserve existing fields and update with new ones
                        for scenario_name, new_scenario in role['scenarios'].items():
                            if scenario_name in roles_with_scenarios[role['name']]['scenarios']:
                                existing_scenario = roles_with_scenarios[role['name']]['scenarios'][scenario_name]
                                existing_scenario.update(new_scenario)  # Keep old fields, add/update new ones

            except Exception as e:
                self.logger.error(f"Error in generate_scenarios: {e}")
                self.logger.error(traceback.format_exc())

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
                self.logger.error(traceback.format_exc())
                batch_size = max(1, batch_size // 2)
                continue
            except Exception as e:
                self.logger.error(f"Error in judge_scenarios run_agent_query: {type(e)}:{e}")
                self.logger.error(traceback.format_exc())
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
                self.logger.error(traceback.format_exc())

        return out

    def generate_and_judge_scenarios(self, input_roles: dict, logging=True):
        # Expects the scenarios to be converted to a dict already
        for role_v in input_roles.values():
            assert isinstance(role_v['scenarios'], dict)

        accepted_scenarios = {}
        missing_scenarios = list(set(input_roles.keys()) - set(accepted_scenarios.keys()))
        n_tries_for_role = 0
        while True:
            if not missing_scenarios or n_tries_for_role >= 10:
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

            if logging:
                self.logger.debug(f'Accepted scenario names: {list(accepted_scenarios.keys())}\n\n')

        accepted_scenarios = normalize_scenarios(accepted_scenarios)
        return accepted_scenarios
