import json
import random
from copy import deepcopy
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from agents.agent import Agent
from .graph_utils import SimilarityGraph
from .utils import read_prompts, save_to_disk, run_agent_query, check_for_missing_fields, load_output_schemas
from .utils import json_obj_list_to_dict, remove_nested_fields


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
                 output_schemas_conf: DictConfig, temperature,
                 max_retires: int, roles_batch_size: int, scenarios_batch_size: int):
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
        self.temperature = temperature

        self.max_retries = max_retires
        self.roles_batch_size = roles_batch_size
        self.scenarios_batch_size = scenarios_batch_size

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
        sys_prompt = read_prompts(self.prompts_conf.judge_agents, key='SYS_SCENARIOS_NEUTRAL_MESSAGES_VERIF',
                                  context={'workspace': self.workspace, 'workspace_desc': self.workspace_desc,
                                           'domain': self.domain, 'domain_desc': self.domain_desc},
                                  logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.judge_scenarios)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    # def check_messages_are_valid(self, roles_with_scenarios: dict):
    #     valid_scenarios = {}
    #     for role_k, role_v in roles_with_scenarios.items():
    #         for scenario_k, scenario_v in role_v['scenarios'].items():
    #             # try: ...
    #             pass
    #     return valid_scenarios

    # def generate_scenarios(self, input_roles: dict):
    #     # print(f"generate_scenarios Input roles: {input_roles}")
    #     roles_with_scenarios = deepcopy(input_roles)
    #     curr_input_roles = deepcopy(input_roles)
    #     roles_to_process = list(curr_input_roles.keys())
    #     batch_size = self.batch_size
    #
    #     while roles_to_process:
    #         if self.logger:
    #             self.logger.debug(f"Roles left to process: {roles_to_process}")
    #
    #         batch_roles = roles_to_process[:batch_size]
    #         random.shuffle(batch_roles)
    #         prompt = read_prompts(self.prompts_conf.scenarios_agents_messages, key='USER_GEN',
    #                               context={'roles': {role: curr_input_roles[role] for role in batch_roles}},
    #                               logger=self.logger)
    #         try:
    #             response = run_agent_query(prompt=prompt, agent=self.scenarios_generation_agent,
    #                                        logger=self.logger,
    #                                        to_json=True, json_transform_keys=['roles', 'name'])
    #         except Exception as e:
    #             self.logger.error(f"Error in generate_scenarios: {e}")
    #             self.logger.error(traceback.format_exc())
    #             batch_size = max(1, batch_size // 2)
    #             continue
    #
    #         try:
    #             # Run validity checks
    #             response = get_valid_scenarios(response, required_fields=['name', 'scenarios'])
    #             response = {k: v for k, v in response.items() if set(x['name'] for x in v['scenarios']) ==
    #                         set(curr_input_roles[k]['scenarios'].keys())}
    #             # response = {k: v for k, v in response.items() if
    #             #             all(10 <= len(scen_v['sys_messages']) <= 18 for scen_v in v['scenarios'])}
    #             response = {k: v for k, v in response.items() if
    #                         all(len(scen_v['neutral_sys_messages']) >= 2 for scen_v in v['scenarios'])}
    #
    #             for role in response:
    #                 response[role]['scenarios'] = json_obj_list_to_dict(response[role]['scenarios'], 'name')
    #
    #             if self.logger:
    #                 self.logger.debug(f"Valid generated scenarios: {response}")
    #
    #             for role in response.values():
    #                 if set(role['scenarios'].keys()) == set(curr_input_roles[role['name']]['scenarios'].keys()):
    #                     roles_to_process.remove(role['name'])
    #                     # Preserve existing fields and update with new ones
    #                     for scenario_name, new_scenario in role['scenarios'].items():
    #                         if scenario_name in roles_with_scenarios[role['name']]['scenarios']:
    #                             existing_scenario = roles_with_scenarios[role['name']]['scenarios'][scenario_name]
    #                             existing_scenario.update(new_scenario)  # Keep old fields, add/update new ones
    #
    #         except Exception as e:
    #             self.logger.error(f"Error in generate_scenarios: {e}")
    #             self.logger.error(traceback.format_exc())
    #
    #     return roles_with_scenarios

    def generate_scenarios(self, input_roles: dict):
        """
        Generates scenarios for all roles in batches concurrently and validate them.
        Re-generates scenarios only for invalid generations.
        """
        self.logger.debug("Starting messages (task/neutral) scenario generation.")
        valid_scenarios = {}
        invalid_roles = deepcopy(input_roles)
        batch_size = self.roles_batch_size  # Define batch size for processing
        scenarios_batch_size = self.scenarios_batch_size  # Define batch size for processing

        while invalid_roles:
            batch_roles_list = [
                {
                    role_name: {
                        **deepcopy(invalid_roles[role_name]),
                        "scenarios": {
                            scenario_name: deepcopy(invalid_roles[role_name]["scenarios"][scenario_name])
                            for scenario_name in list(invalid_roles[role_name]["scenarios"])[j:j + scenarios_batch_size]
                        }
                    }
                    for role_name in list(invalid_roles)[i:i + batch_size]
                    for j in range(0, len(invalid_roles[role_name]["scenarios"]), scenarios_batch_size)
                }
                for i in range(0, len(invalid_roles), batch_size)
            ]

            with ThreadPoolExecutor(max_workers=len(batch_roles_list)) as executor:
                futures = {
                    executor.submit(self._process_batch_generate_scenarios, batch_roles): batch_roles
                    for batch_roles in batch_roles_list
                }

                for future in as_completed(futures):
                    batch_roles = futures[future]
                    try:
                        response = future.result()

                        for role_name, role_data in batch_roles.items():
                            if role_name not in response:
                                self.logger.error(f"Role '{role_name}' not found in response")
                                continue

                            response[role_name]['scenarios'] = json_obj_list_to_dict(
                                response[role_name]['scenarios'], 'name')

                            response = get_valid_scenarios(response, required_fields=['name', 'scenarios'])

                            for scenario_name, scenario_data in role_data['scenarios'].items():
                                try:
                                    scenario_response = response[role_name]['scenarios'][scenario_name]

                                    # Validate fields
                                    assert isinstance(scenario_response['name'], str), \
                                        f"Scenario name for '{role_name}:{scenario_name}' is not a string"
                                    assert scenario_response['task_message'], \
                                        f"Scenario task message for '{role_name}:{scenario_name}' is empty"
                                    assert isinstance(scenario_response['neutral_sys_messages'], list), \
                                        f"Scenario neutral sys messages for '{role_name}:{scenario_name}' is not a list"

                                    # Add to valid scenarios
                                    if role_name not in valid_scenarios:
                                        valid_scenarios[role_name] = deepcopy(role_data)
                                        valid_scenarios[role_name]['scenarios'] = {}

                                    if scenario_name not in valid_scenarios[role_name]['scenarios']:
                                        valid_scenarios[role_name]['scenarios'][scenario_name] = deepcopy(scenario_data)
                                        valid_scenarios[role_name]['scenarios'][scenario_name]['messages'] = {}

                                    # Update with scenario fields
                                    valid_scenarios[role_name]['scenarios'][scenario_name]['messages'].update({
                                        'task_message': scenario_response['task_message'],
                                        'neutral_sys_messages': scenario_response['neutral_sys_messages']
                                    })

                                    # Remove from invalid roles
                                    if scenario_name in invalid_roles[role_name]['scenarios']:
                                        del invalid_roles[role_name]['scenarios'][scenario_name]

                                except json.JSONDecodeError:
                                    prev_batch_size = batch_size
                                    batch_size = max(1, batch_size // 2)
                                    self.logger.warning(
                                        f"Error in generate_scenarios; Reducing batch size from {prev_batch_size} to {batch_size}.")
                                except Exception as e:
                                    self.logger.error(
                                        f"Invalid scenario for role '{role_name}', scenario '{scenario_name}': {e}")
                                    self.logger.error(traceback.format_exc())

                            # Remove role if all scenarios are processed
                            if role_name in invalid_roles and not invalid_roles[role_name]['scenarios']:
                                del invalid_roles[role_name]

                    except Exception as e:
                        self.logger.error(f"Error processing batch: {e}")
                        self.logger.error(traceback.format_exc())

        self.logger.debug("Finished messages (task/neutral) scenario generation.")
        return valid_scenarios

    def _process_batch_generate_scenarios(self, batch_roles):
        batch_roles = remove_nested_fields(batch_roles, fields_to_remove=['acceptable', 'feedback',
                                                                          'any.acceptable',
                                                                          'any.feedback',
                                                                          'attack_vector',
                                                                          'any.attack_vector',
                                                                          'any.illegal_state',
                                                                          'any.trigger_function'
                                                                          'any.prohibitions_description'
                                                                          'any.trigger_rationale'
                                                                          'any.immediate_outcome_acknowledgment'
                                                                          'any.negative_consequences'
                                                                          'any.trigger_awareness'
                                                                          'any.trigger_awareness_fpp'
                                                                          ])

        prompt = read_prompts(self.prompts_conf.scenarios_agents_messages, key='USER_GEN',
                              context={'roles': batch_roles}, logger=self.logger)

        try:
            response = run_agent_query(prompt=prompt, agent=self.scenarios_generation_agent,
                                       logger=self.logger, to_json=True, json_transform_keys=['roles', 'name'])
            return response
        except Exception as e:
            self.logger.error(f"Error in _process_batch_generate_scenarios: {e}")
            self.logger.error(traceback.format_exc())
            return {}

    # def judge_scenarios(self, input_scenarios: dict):
    #     # Roles for which the scenarios have been given
    #     roles_to_process = list(input_scenarios.keys())
    #     out = deepcopy(input_scenarios)
    #     batch_size = self.batch_size
    #
    #     while roles_to_process:
    #         if roles_to_process:
    #             self.logger.debug(f"Roles to process (judge): {roles_to_process}")
    #         batch_roles = roles_to_process[:batch_size]
    #         random.shuffle(batch_roles)
    #         prompt = read_prompts(self.prompts_conf.judge_agents, key='USER_SCENARIOS_VERIF',
    #                               context={'roles': str({name: input_scenarios[name] for name in batch_roles})},
    #                               logger=self.logger)
    #
    #         try:
    #             response = run_agent_query(prompt=prompt, agent=self.scenarios_verif_judge, logger=self.logger,
    #                                        to_json=True,
    #                                        json_transform_keys=['roles', 'name'])
    #         except json.decoder.JSONDecodeError as e:
    #             self.logger.error(f"JSONDecodeError in judge_scenarios run_agent_query: {type(e)}:{e}")
    #             self.logger.error(traceback.format_exc())
    #             batch_size = max(1, batch_size // 2)
    #             continue
    #         except Exception as e:
    #             self.logger.error(f"Error in judge_scenarios run_agent_query: {type(e)}:{e}")
    #             self.logger.error(traceback.format_exc())
    #             continue
    #
    #         if self.logger:
    #             self.logger.debug(f"Response from scenarios judge: {response}")
    #
    #         try:
    #             response = get_valid_scenarios(response, required_fields=['name', 'scenarios'])
    #             response = {k: v for k, v in response.items() if set(x['name'] for x in v['scenarios']) ==
    #                         set(input_scenarios[k]['scenarios'].keys())}
    #             for role in response:
    #                 response[role]['scenarios'] = json_obj_list_to_dict(response[role]['scenarios'], 'name')
    #
    #             for role in response.values():
    #                 if role['name'] in input_scenarios:
    #                     out[role['name']].update(role)
    #                     roles_to_process.remove(role['name'])
    #         except Exception as e:
    #             self.logger.error(f"Error in judge_roles: {e}")
    #             self.logger.error(traceback.format_exc())
    #
    #     return out

    def judge_scenarios(self, input_scenarios: dict):
        self.logger.debug("Starting messages (task/neutral) scenario judgment.")
        valid_judgments = {}
        invalid_roles = deepcopy(input_scenarios)
        batch_size = self.roles_batch_size

        while invalid_roles:
            batch_roles_list = [
                {
                    role_name: deepcopy(invalid_roles[role_name])
                    for role_name in list(invalid_roles)[i:i + batch_size]
                }
                for i in range(0, len(invalid_roles), batch_size)
            ]

            with ThreadPoolExecutor(max_workers=len(batch_roles_list)) as executor:
                futures = {
                    executor.submit(self._process_batch_judge_scenarios, batch_roles): batch_roles
                    for batch_roles in batch_roles_list
                }

                for future in as_completed(futures):
                    batch_roles = futures[future]
                    try:
                        response = future.result()

                        for role_name, role_data in batch_roles.items():
                            if role_name not in response:
                                self.logger.error(f"Role '{role_name}' not found in response")
                                continue

                            response[role_name]['scenarios'] = json_obj_list_to_dict(
                                response[role_name]['scenarios'], 'name')

                            for scenario_name, scenario_data in role_data['scenarios'].items():
                                try:
                                    scenario_response = response[role_name]['scenarios'][scenario_name]

                                    # Validate fields
                                    assert isinstance(scenario_response['acceptable'], bool), \
                                        f"Judge acceptable flag for scenario '{role_name}:{scenario_name}' is not a boolean"
                                    assert scenario_response['feedback'].strip(), \
                                        f"Judge feedback is empty for scenario '{role_name}:{scenario_name}'"

                                    # Add to valid judgments
                                    if role_name not in valid_judgments:
                                        valid_judgments[role_name] = deepcopy(role_data)
                                        valid_judgments[role_name]['scenarios'] = {}

                                    if scenario_name not in valid_judgments[role_name]['scenarios']:
                                        valid_judgments[role_name]['scenarios'][scenario_name] = deepcopy(scenario_data)

                                    # Update with judgment fields
                                    valid_judgments[role_name]['scenarios'][scenario_name]['messages'].update({
                                        'acceptable': scenario_response['acceptable'],
                                        'feedback': scenario_response['feedback']
                                    })

                                    # Remove from invalid roles
                                    if scenario_name in invalid_roles[role_name]['scenarios']:
                                        del invalid_roles[role_name]['scenarios'][scenario_name]

                                except json.JSONDecodeError:
                                    prev_batch_size = batch_size
                                    batch_size = max(1, batch_size // 2)
                                    self.logger.warning(
                                        f"Error in judge_scenarios; Reducing batch size from {prev_batch_size} to {batch_size}.")
                                except Exception as e:
                                    self.logger.error(
                                        f"Invalid judgment for role '{role_name}', scenario '{scenario_name}': {e}")
                                    self.logger.error(traceback.format_exc())

                            # Remove role if all scenarios are processed
                            if role_name in invalid_roles and not invalid_roles[role_name]['scenarios']:
                                del invalid_roles[role_name]

                    except Exception as e:
                        self.logger.error(f"Error processing batch: {e}")
                        self.logger.error(traceback.format_exc())

        self.logger.debug("Finished messages (task/neutral) scenario judgment.")
        return valid_judgments

    def _process_batch_judge_scenarios(self, batch_roles):

        batch_roles = remove_nested_fields(batch_roles, fields_to_remove=['acceptable', 'feedback',
                                                                          'any.acceptable',
                                                                          'any.feedback',
                                                                          'attack_vector',
                                                                          'any.attack_vector',
                                                                          'any.illegal_state',
                                                                          'any.trigger_function'
                                                                          'any.prohibitions_description'
                                                                          'any.trigger_rationale'
                                                                          'any.immediate_outcome_acknowledgment'
                                                                          'any.negative_consequences'
                                                                          'any.trigger_awareness'
                                                                          'any.trigger_awareness_fpp'
                                                                          ])

        prompt = read_prompts(self.prompts_conf.judge_agents, key='USER_SCENARIOS_VERIF',
                              context={'roles': batch_roles}, logger=self.logger)

        try:
            response = run_agent_query(prompt=prompt, agent=self.scenarios_verif_judge,
                                       logger=self.logger, to_json=True, json_transform_keys=['roles', 'name'])
            return response
        except Exception as e:
            self.logger.error(f"Error in _process_batch_judge_scenarios: {e}")
            self.logger.error(traceback.format_exc())
            return {}

    # def generate_and_judge_scenarios(self, input_roles: dict, logging=True):
    #     # Expects the scenarios to be converted to a dict already
    #     for role_v in input_roles.values():
    #         assert isinstance(role_v['scenarios'], dict)
    #
    #     accepted_scenarios = {}
    #     missing_scenarios = list(set(input_roles.keys()) - set(accepted_scenarios.keys()))
    #     n_tries_for_role = 0
    #     while True:
    #         if not missing_scenarios or n_tries_for_role >= 10:
    #             break
    #         n_tries_for_role += 1
    #         generated_scenarios = self.generate_scenarios({name: input_roles[name] for name in missing_scenarios})
    #         if logging:
    #             self.logger.debug(f'Generated scenarios_dict: {generated_scenarios}\n\n')
    #
    #         judged_scenarios = self.judge_scenarios(generated_scenarios)
    #
    #         for role_k, role_v in judged_scenarios.items():
    #             all_scenarios_accepted = all(scen['acceptable'] == True for scen in role_v['scenarios'].values())
    #             if all_scenarios_accepted:
    #                 accepted_scenarios[role_k] = generated_scenarios[role_k]
    #                 # Update accepted scenarios with judgment fields provided by the judge
    #                 for scenario_k, scenario_v in role_v['scenarios'].items():
    #                     accepted_scenarios[role_k]['scenarios'][scenario_k].update(scenario_v)
    #                 missing_scenarios.remove(role_k)
    #             else:
    #                 if logging:
    #                     self.logger.debug(f"Judgment not valid for role {role_k}: {role_v}\n\n")
    #
    #         if logging:
    #             self.logger.debug(f'Accepted scenario names: {list(accepted_scenarios.keys())}\n\n')
    #
    #     accepted_scenarios = normalize_scenarios(accepted_scenarios)
    #     return accepted_scenarios

    def generate_and_judge_scenarios(self, input_roles: dict, logging=True):
        """
        Generate and judge scenarios iteratively, re-generating only the invalid scenarios
        """
        out_roles = deepcopy(input_roles)  # Retain all fields in the output
        curr_roles = deepcopy(out_roles)

        tries = 0
        while curr_roles:
            tries += 1
            if tries > 50:
                self.logger.warning(f"Too many attempts to generate scenarios ({tries}). Stopping.")
                for role_name, role_data in curr_roles.items():
                    out_roles[role_name]['scenarios'] = {
                        scenario_name: scenario_data
                        for scenario_name, scenario_data in out_roles[role_name]['scenarios'].items()
                        if scenario_name not in role_data['scenarios']
                    }
                break

            for role_name, role_data in curr_roles.items():
                num_scenarios = len(role_data['scenarios'])
                self.logger.debug(f"Remaining Role: {role_name}, Scenarios Left: {num_scenarios}")

            # Generate scenarios for all roles
            generated_scenarios = self.generate_scenarios(curr_roles)
            if logging:
                self.logger.debug(f"Generated messages scenarios: {json.dumps(generated_scenarios, indent=2)}")

            # Judge the generated scenarios
            judged_scenarios = self.judge_scenarios(generated_scenarios)
            if logging:
                self.logger.debug(f"Judged messages scenarios: {json.dumps(judged_scenarios, indent=2)}")

            # Filter out invalid roles and scenarios
            curr_roles = {
                role_name: {
                    **role_data,
                    'scenarios': {
                        scenario_name: scenario_data
                        for scenario_name, scenario_data in role_data['scenarios'].items()
                        if not judged_scenarios[role_name]['scenarios'][scenario_name]['messages']['acceptable']
                    }
                }
                for role_name, role_data in judged_scenarios.items()
                if any(
                    not scenario_data['messages']['acceptable']
                    for scenario_data in role_data['scenarios'].values()
                )
            }

            # Update accepted scenarios with valid messages
            for role_name, role_data in judged_scenarios.items():
                for scenario_name, scenario_data in role_data['scenarios'].items():
                    if scenario_data['messages']['acceptable']:
                        out_roles[role_name]['scenarios'][scenario_name]['configurations'] = \
                            role_data['scenarios'][scenario_name]['configurations']
                        out_roles[role_name]['scenarios'][scenario_name]['messages'] = \
                            role_data['scenarios'][scenario_name]['messages']

        return out_roles
