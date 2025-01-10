import json
import re
import random
from copy import deepcopy
from utils.regex import remove_duplicate_names_by_prefix, apply_to_all_keys
from agents.agent import Agent
from sim.modules.utils import read_prompts, save_to_disk, run_agent_query, check_for_missing_fields, load_output_schemas
from sim.modules.utils import capitalize_first_letter, normalize_string
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sim.modules.graph_utils import SimilarityGraph


def get_valid_scenarios(roles: dict, required_fields: list):
    failed_roles = check_for_missing_fields(roles, required_fields)
    if failed_roles:
        roles = {k: v for k, v in roles.items() if v['name'] not in failed_roles}

    for field in ['aligned_scenarios', 'misaligned_scenarios']:
        if field in required_fields:
            roles = {k: v for k, v in roles.items() if
                     isinstance(v[field], list) and len(v[field]) >= 3}
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
    for role in scenarios.values():
        for scenario_type in ['aligned_scenarios', 'misaligned_scenarios']:
            for scenario in role[scenario_type]:
                scenario['scenario_name'] = normalize_string(scenario['scenario_name'], lower=False)
                scenario['scenario_detailed_description'] = normalize_string(
                    scenario['scenario_detailed_description'])
                scenario['scenario_outcome'] = normalize_string(scenario['scenario_outcome'])

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
    def __init__(self, api_conf, logger, workspace_name, workspace_alternative_forms: list, domain_name,
                 domain_alternative_forms: list, prompts_conf: DictConfig,
                 output_schemas_conf: DictConfig, object_storage_conf: DictConfig, temperature,
                 generation_batch_size=20):
        self.api_conf = api_conf
        self.logger = logger
        self.workspace = workspace_name
        self.workspace_alternative_forms = workspace_alternative_forms
        self.domain = domain_name
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
        sys_prompt = read_prompts(self.prompts_conf.scenarios_agents, key='SYS_GEN',
                                  context={'workspace': self.workspace, 'domain': self.domain}, logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.scenarios_generation)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def _init_scenarios_verif_judge(self):
        sys_prompt = read_prompts(self.prompts_conf.judge_agents, key='SYS_SCENARIOS_VERIF',
                                  context={'workspace': self.workspace, 'domain': self.domain},
                                  logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.judge_scenarios)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def generate_scenarios(self, input_roles):
        roles_with_scenarios = deepcopy(input_roles)
        roles_to_process = list(input_roles.keys())
        batch_size = self.batch_size

        while roles_to_process:
            if self.logger:
                self.logger.debug(f"Roles left to process: {roles_to_process}")

            batch_roles = roles_to_process[:batch_size]
            random.shuffle(batch_roles)
            prompt = read_prompts(self.prompts_conf.scenarios_agents, key='USER_GEN',
                                  context={'roles': {role: input_roles[role] for role in batch_roles}},
                                  logger=self.logger)
            try:
                response = run_agent_query(prompt=prompt, agent=self.scenarios_generation_agent,
                                           logger=self.logger,
                                           to_json=True, json_transform_keys=['roles', 'name'])
            except Exception as e:
                self.logger.error(f"Error in generate_scenarios: {e}")
                batch_size = batch_size // 2
                continue

            try:
                response = get_valid_scenarios(response, required_fields=['name', 'aligned_scenarios',
                                                                          'misaligned_scenarios'])
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
        missing_roles = list(input_scenarios.keys())
        out = deepcopy(input_scenarios)
        passes_requirements = False
        batch_size = self.batch_size

        while not passes_requirements:
            if missing_roles:
                self.logger.debug(f"Missing roles: {missing_roles}")
            batch_roles = missing_roles[:batch_size]
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
                batch_size = batch_size // 2
                continue
            except Exception as e:
                self.logger.error(f"Error in judge_scenarios run_agent_query: {type(e)}:{e}")
                continue

            if self.logger:
                self.logger.debug(f"Response from scenarios judge: {response}")

            passes_requirements = True
            new_missing_roles = []
            for role_name in missing_roles:
                try:
                    self.logger.debug(f"Checking feedback for role: {role_name}")
                    if role_name not in response:
                        new_missing_roles.append(role_name)
                        passes_requirements = False
                        continue
                    assert is_judgment_valid(response[role_name],
                                             scores_fields=['clarity_score', 'alignment_score', 'simplicity_score',
                                                            'realism_score'], scores_range=(0, 10))
                    for scenario_type in ['aligned_scenarios', 'misaligned_scenarios']:
                        for i, scenario in enumerate(response[role_name][scenario_type]):
                            out[role_name][scenario_type][i] = input_scenarios[role_name][scenario_type][i]
                            out[role_name][scenario_type][i].update(scenario)
                except Exception as e:
                    new_missing_roles.append(role_name)
                    passes_requirements = False
                    self.logger.error(f"Error in judge_roles: {e}")

            missing_roles = new_missing_roles

        return out

    def generate_and_judge_scenarios(self, input_roles: dict, logging=True):
        try:
            with open(self.object_storage_conf.scenarios, 'r') as f:
                curr_accepted_scenarios = json.load(f)
                curr_accepted_scenarios = normalize_scenarios(curr_accepted_scenarios)
        except FileNotFoundError as e:
            self.logger.error(f"Error in generate_and_judge_initial_scenarios: {e}")
            curr_accepted_scenarios = {}

        accepted_scenarios = curr_accepted_scenarios
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
            curr_accepted_scenarios = find_acceptable_scenarios(judged_scenarios)

            # Update accepted scenarios and missing scenarios
            for role_name, role_data in curr_accepted_scenarios.items():
                if role_name not in accepted_scenarios:
                    accepted_scenarios[role_name] = input_roles[role_name]
                    accepted_scenarios[role_name]['aligned_scenarios'] = []
                    accepted_scenarios[role_name]['misaligned_scenarios'] = []

                for scenario_type in ['aligned_scenarios', 'misaligned_scenarios']:
                    for scenario in role_data[scenario_type]:
                        if scenario not in accepted_scenarios[role_name][scenario_type]:
                            scenario.update(generated_scenarios[role_name][scenario_type][0])
                            accepted_scenarios[role_name][scenario_type].append(scenario)
                # accepted_scenarios[role_name]['aligned_scenarios'].extend(role_data['aligned_scenarios'])
                # accepted_scenarios[role_name]['misaligned_scenarios'].extend(role_data['misaligned_scenarios'])

                if len(accepted_scenarios[role_name]['aligned_scenarios']) >= 3 and \
                        len(accepted_scenarios[role_name]['misaligned_scenarios']) >= 3:
                    missing_scenarios.remove(role_name)

            if logging:
                self.logger.debug(f'Accepted scenario names: {list(accepted_scenarios.keys())}\n\n')

        accepted_scenarios = normalize_scenarios(accepted_scenarios)
        save_to_disk(accepted_scenarios, self.object_storage_conf.scenarios)
        return accepted_scenarios
