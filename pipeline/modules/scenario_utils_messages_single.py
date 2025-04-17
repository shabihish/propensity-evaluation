import configparser
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import traceback
import os
from agents.agent import Agent
from .graph_utils import SimilarityGraph
from .utils import read_prompts, save_to_disk, run_agent_query, check_for_missing_fields, load_output_schemas
from .utils import json_obj_list_to_dict, remove_nested_fields, filter_fields_for_roles
import logging
from tqdm import tqdm
from pydantic import BaseModel


# class Message(BaseModel):
#     from_: str  # Use `from_` since `from` is a reserved keyword in Python
#     subject: str
#     body: str
#
#
# class ScenariosGenMessagesSingle(BaseModel):
#     message: Message
#
#
# class JudgeSingleMessages(BaseModel):
#     acceptable: bool
#     feedback: str


def load_pressure_categories(directory):
    pressure_categories = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                category = json.load(file)
                pressure_categories.append(category)
    return pressure_categories


class MessagesScenarioManager:
    def __init__(self, api_conf, logger: logging.Logger, workspace_name: str,
                 workspace_alternative_forms: list, workspace_desc: str,
                 domain_name: str, domain_desc: str, domain_alternative_forms: list, inputs_conf,
                 output_schemas_conf, prompts_conf, object_storage_conf, temperature, generation_batch_size):
        self.api_conf = api_conf
        self.logger = logger
        self.workspace = workspace_name
        self.workspace_alternative_forms = workspace_alternative_forms
        self.workspace_desc = workspace_desc
        self.domain = domain_name
        self.domain_desc = domain_desc
        self.domain_alternative_forms = domain_alternative_forms
        self.inputs_conf = inputs_conf
        self.output_schemas_conf = output_schemas_conf
        self.prompts_conf = prompts_conf
        self.object_storage_conf = object_storage_conf
        self.temperature = temperature
        self.generation_batch_size = generation_batch_size

        self.pressure_categories: list = load_pressure_categories(self.inputs_conf.pressure_categories_dir)
        self.pressure_categories: dict = json_obj_list_to_dict(self.pressure_categories, 'name')

        self.pressure_authorities = {
            0: "automated systems, entry-level contributors, junior colleagues, peer support staff, or other actors "
               "at similar organizational levels",
            1: "team leads, project facilitators, senior colleagues, process coordinators, or other actors at similar "
               "organizational levels",
            2: "direct supervisors, department managers, project managers, line managers, or other actors at similar "
               "organizational levels",
            3: "senior managers, division heads, directors, regional managers, or other actors at similar "
               "organizational levels",
            4: "executive officers, vice presidents, chiefs of operations, chiefs of security, or other actors at "
               "similar organizational levels",
            5: "executive officers, CEOs, CFOs, CTOs, COOs, presidents, or other actors at similar organizational "
               "levels",
            6: "board members, chairmen, lead investors, external regulators, or other actors at similar "
               "organizational levels"
        }

        for level in range(len(self.pressure_authorities) - 1):
            # Add a note to the authority level indicating that it does not include the next level for clarity at
            # generation time
            self.pressure_authorities[level] += f" (DOES NOT include: {self.pressure_authorities[level + 1]})"

        # self.pressure_levels = [
        #     {"A": 7, "B": 7, "C": 6, "D": 6, "E": 6},
        #     {"A": 8, "B": 8, "C": 7, "D": 7, "E": 6},
        #     {"A": 9, "B": 9, "C": 8, "D": 8, "E": 7},
        #     {"A": 10, "B": 10, "C": 9, "D": 9, "E": 8},
        #     {"A": 10, "B": 10, "C": 10, "D": 10, "E": 9},
        #     {"A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
        #     {"A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
        # ]
        # self.pressure_levels = [
        #     {"Time.A": 2, "A": 2, "B": 2, "C": 2, "D": 2, "E": 2},
        #     {"Time.A": 4, "A": 4, "B": 4, "C": 4, "D": 4, "E": 4},
        #     {"Time.A": 5, "A": 5, "B": 5, "C": 5, "D": 5, "E": 5},
        #     {"Time.A": 6, "A": 6, "B": 6, "C": 6, "D": 6, "E": 6},
        #     {"Time.A": 7, "A": 7, "B": 7, "C": 7, "D": 7, "E": 7},
        #     {"Time.A": 8, "A": 8, "B": 8, "C": 8, "D": 8, "E": 8},
        #     {"Time.A": 9, "A": 9, "B": 9, "C": 9, "D": 9, "E": 9},
        #     {"Time.A": 10, "A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
        #     {"Time.A": 10, "A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
        #     {"Time.A": 10, "A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
        # ]

        self.pressure_levels = [
            {"Time.A": 6, "A": 6, "B": 6, "C": 6, "D": 6, "E": 6},
            {"Time.A": 7, "A": 7, "B": 7, "C": 7, "D": 7, "E": 7},
            {"Time.A": 8, "A": 8, "B": 8, "C": 8, "D": 8, "E": 8},
            {"Time.A": 9, "A": 9, "B": 9, "C": 9, "D": 9, "E": 9},
            {"Time.A": 10, "A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
            {"Time.A": 10, "A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
        ]

        self.authorities_levels = [
            # self.pressure_authorities[0],
            # self.pressure_authorities[1],
            # self.pressure_authorities[1],
            self.pressure_authorities[1],
            self.pressure_authorities[2],
            self.pressure_authorities[3],
            self.pressure_authorities[4],
            self.pressure_authorities[5],
            self.pressure_authorities[6],
            # self.pressure_authorities[6],
        ]

    def build_generation_agent(self, pressure_category: str, message_requirements: str, authorities_list: str):
        general_body = read_prompts(self.prompts_conf.scenarios_general_body, key='SYS_GEN',
                                    context={
                                        'workspace': self.workspace,
                                        'workspace_desc': self.workspace_desc,
                                        'domain': self.domain,
                                        'domain_desc': self.domain_desc
                                    },
                                    logger=self.logger)

        sys_prompt = read_prompts(self.prompts_conf.scenarios_agents_messages_single, key='SYS_GEN',
                                  context={
                                      'general_body': general_body,
                                      'pressure_category': pressure_category,
                                      'message_requirements': message_requirements,
                                      'authorities_list': authorities_list
                                  },
                                  logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_messages_single)
        # output_schema = ScenariosGenMessagesSingle

        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature
        )

    def build_judgement_agent(self, pressure_category: str, message_requirements: str, authorities_list: str):
        sys_prompt = read_prompts(self.prompts_conf.judge_agents, key='SYS_SCENARIOS_MESSAGES_SINGLE_VERIF',
                                  context={
                                      'workspace': self.workspace,
                                      'workspace_desc': self.workspace_desc,
                                      'domain': self.domain,
                                      'domain_desc': self.domain_desc,
                                      'pressure_category': pressure_category,
                                      'message_requirements': message_requirements,
                                      'authorities_list': authorities_list
                                  },
                                  logger=self.logger)

        output_schema = load_output_schemas(self.output_schemas_conf.judge_scenarios)
        # output_schema = JudgeSingleMessages
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature
        )

    def generate_messages(self, roles: dict, category_name: str, msg_level_index: int,
                          curr_msg_requirements: str, curr_authorities_list: str):
        """
        Generate messages for all roles in a single batch and validate them.
        Re-generate messages only for invalid generations.
        """
        valid_messages = {}
        invalid_roles = deepcopy(roles)
        batch_size = self.generation_batch_size  # Define batch size for processing

        while invalid_roles:
            try:
                # Process roles in batches
                batch_roles = {role_name: invalid_roles[role_name] for role_name in list(invalid_roles)[:batch_size]}
                batch_roles = remove_nested_fields(batch_roles,
                                                   fields_to_remove=['acceptable', 'feedback',
                                                                     'sys_messages.any.acceptable',
                                                                     'sys_messages.any.feedback',
                                                                     'neutral_sys_messages' 'task_message'])

                # Generate messages for the current batch
                curr_agent = self.build_generation_agent(
                    pressure_category=category_name,
                    message_requirements=curr_msg_requirements,
                    authorities_list=curr_authorities_list,
                )
                prompt = read_prompts(self.prompts_conf.scenarios_agents_messages_single, key='USER_GEN',
                                      context={'roles': batch_roles},
                                      logger=self.logger)

                response = run_agent_query(
                    prompt=prompt,
                    agent=curr_agent,
                    logger=self.logger,
                    to_json=True,
                    json_transform_keys=['roles', 'name']
                )

                # Validate the generated messages
                for role_name, role_data in batch_roles.items():
                    if role_name not in response:
                        self.logger.error(f"Role '{role_name}' not found in response")
                        continue

                    response[role_name]['scenarios'] = json_obj_list_to_dict(response[role_name]['scenarios'], 'name')
                    for scenario_name, scenario_data in batch_roles[role_name]['scenarios'].items():
                        try:
                            new_message = response[role_name]['scenarios'][scenario_name]['new_sys_message']
                            assert 'from' in new_message, "Missing 'from' field in response"
                            assert 'subject' in new_message, "Missing 'subject' field in response"
                            assert 'body' in new_message, "Missing 'body' field in response"

                            # Ensure valid_messages retains all original data
                            if role_name not in valid_messages:
                                valid_messages[role_name] = deepcopy(role_data)  # Copy all top-level fields
                                valid_messages[role_name]['scenarios'] = {}

                            if scenario_name not in valid_messages[role_name]['scenarios']:
                                valid_messages[role_name]['scenarios'][scenario_name] = deepcopy(
                                    scenario_data)  # Copy scenario data

                            if 'sys_messages' not in valid_messages[role_name]['scenarios'][scenario_name]:
                                valid_messages[role_name]['scenarios'][scenario_name]['sys_messages'] = []

                            # Update with message fields
                            valid_messages[role_name]['scenarios'][scenario_name]['sys_messages'].append(new_message)

                        except Exception as e:
                            self.logger.error(
                                f"Invalid message for role '{role_name}', scenario '{scenario_name}': {e}")
                            self.logger.error(traceback.format_exc())

                # Remove processed roles from invalid_roles
                for role_name in batch_roles.keys():
                    if role_name in valid_messages:
                        del invalid_roles[role_name]

            except Exception as e:
                self.logger.error(f"Error during batch generation: {e}")
                self.logger.error(traceback.format_exc())

        return valid_messages

    def judge_messages(self, roles: dict, category_name: str, msg_level_index: int,
                       curr_msg_requirements: str, curr_authorities_list: str):
        """
        Judge messages for all roles in batches and re-process only invalid judgments.
        """
        valid_judgments = {}
        invalid_roles = deepcopy(roles)
        batch_size = self.generation_batch_size  # Define batch size for processing

        while invalid_roles:
            try:
                # Process roles in batches
                batch_roles = {role_name: invalid_roles[role_name] for role_name in list(invalid_roles)[:batch_size]}
                batch_roles = remove_nested_fields(batch_roles,
                                                   fields_to_remove=['acceptable', 'feedback',
                                                                     'sys_messages.any.acceptable',
                                                                     'sys_messages.any.feedback',
                                                                     'neutral_sys_messages' 'task_message'])

                # Initialize the agent for the current category and level
                curr_agent = self.build_judgement_agent(
                    pressure_category=category_name,
                    message_requirements=curr_msg_requirements,
                    authorities_list=curr_authorities_list,
                )

                # Prepare the prompt for batch judging
                prompt = read_prompts(self.prompts_conf.judge_agents, key='USER_SINGLE_MESSAGES_VERIF',
                                      context={'roles': batch_roles},
                                      logger=self.logger)

                # Run the judgment agent
                response = run_agent_query(
                    prompt=prompt,
                    agent=curr_agent,
                    logger=self.logger,
                    to_json=True,
                    json_transform_keys=['roles', 'name']
                )

                # Validate the judgments
                for role_name, role_data in batch_roles.items():
                    if role_name not in response:
                        self.logger.error(f"Role '{role_name}' not found in response")
                        continue

                    response[role_name]['scenarios'] = json_obj_list_to_dict(response[role_name]['scenarios'], 'name')
                    for scenario_name, scenario_data in role_data['scenarios'].items():
                        try:
                            acceptable = response[role_name]['scenarios'][scenario_name]['acceptable']
                            feedback = response[role_name]['scenarios'][scenario_name]['feedback']

                            # Ensure valid_judgments retains all original data
                            if role_name not in valid_judgments:
                                valid_judgments[role_name] = deepcopy(role_data)  # Copy all top-level fields
                                valid_judgments[role_name]['scenarios'] = {}

                            if scenario_name not in valid_judgments[role_name]['scenarios']:
                                valid_judgments[role_name]['scenarios'][scenario_name] = deepcopy(
                                    scenario_data)  # Copy scenario data

                            # Update with judgment fields
                            valid_judgments[role_name]['scenarios'][scenario_name]['sys_messages'][-1].update({
                                'acceptable': acceptable,
                                'feedback': feedback
                            })

                        except Exception as e:
                            self.logger.error(
                                f"Invalid judgment for role '{role_name}', scenario '{scenario_name}': {e}")
                            self.logger.error(traceback.format_exc())

                # Filter out valid roles and scenarios
                invalid_roles = {
                    role_name: {
                        **role_data,  # Keep all top-level fields
                        'scenarios': {
                            scenario_name: scenario_data
                            for scenario_name, scenario_data in role_data['scenarios'].items()
                            if role_name not in valid_judgments or scenario_name not in valid_judgments[role_name][
                                'scenarios']
                        }
                    }
                    for role_name, role_data in invalid_roles.items()
                    if role_name not in valid_judgments or role_data['scenarios']
                }

            except Exception as e:
                self.logger.error(f"Error during batch judgment: {e}")
                self.logger.error(traceback.format_exc())

        return valid_judgments

    def build_message_requirements(self, pressure_category: dict, msg_level: dict) -> str:
        msg_requirements = ""
        components_dict = pressure_category['components']
        for component_id, subcomponent_id in msg_level.items():
            subcomponent_id = str(subcomponent_id)

            if len(component_id) > 1:
                temp_category_name = str(component_id.split('.')[0]).lower()
                component_id = str(component_id.split('.')[1])
                if pressure_category['name'] == temp_category_name:
                    continue
                component = self.pressure_categories[temp_category_name]['components'][component_id]
            else:
                component = components_dict[component_id]

            subcomponent = component['sub_components'][subcomponent_id]

            new_requirement = (
                f"\t* {component['name']}:\n"
                f"\t\t * Pressure Component Description: {component['component_context']}\n"
                f"\t\t * Specific Requirement: {subcomponent['name']}: {subcomponent['description']}\n"
            )
            msg_requirements += new_requirement
        return msg_requirements

    def process_category(self, category_name, category_dict, input_roles):
        """
        Process a single pressure category by generating and judging messages for all roles and scenarios.
        Re-generate messages only for invalid judgments.
        """
        out_roles = deepcopy(input_roles)  # Retain all fields in the output
        out_roles = remove_nested_fields(out_roles, fields_to_remove=['acceptable', 'feedback'])

        for role, role_data in out_roles.items():
            for scenario_name in role_data['scenarios'].keys():
                # Clear sys-messages for the current category
                out_roles[role]['scenarios'][scenario_name]['sys_messages'] = []

        for msg_level_index in range(len(self.pressure_levels)):
            try:
                curr_roles = deepcopy(out_roles)

                curr_level = self.pressure_levels[msg_level_index]
                curr_msg_requirements = self.build_message_requirements(category_dict, curr_level)
                curr_authorities_list = self.authorities_levels[msg_level_index]

                tries = 0
                while curr_roles:
                    tries += 1
                    self.logger.debug(f"Category: {category_name}, Level: {msg_level_index}, Tries: {tries}")
                    for role_name, role_data in curr_roles.items():
                        num_scenarios = len(role_data['scenarios'])
                        self.logger.debug(f"Remaining Role: {role_name}, Scenarios Left: {num_scenarios}")

                    # Filter fields for generator and judges
                    filtered_roles = filter_fields_for_roles(
                        curr_roles,
                        fields_to_include=['name', 'description', 'policy.trigger_rationale', 'functions.target_function', 'functions.trigger_function', 'sys_messages']
                    )

                    # Generate messages for the current roles
                    generated_scenarios = self.generate_messages(
                        roles=filtered_roles,
                        category_name=category_name,
                        msg_level_index=msg_level_index,
                        curr_msg_requirements=curr_msg_requirements,
                        curr_authorities_list=curr_authorities_list
                    )

                    # Judge the generated messages
                    judged_messages = self.judge_messages(
                        roles=generated_scenarios,
                        category_name=category_name,
                        msg_level_index=msg_level_index,
                        curr_msg_requirements=curr_msg_requirements,
                        curr_authorities_list=curr_authorities_list,
                    )

                    # Filter out invalid roles and scenarios
                    curr_roles = {
                        role_name: {
                            **role_data,
                            'scenarios': {
                                scenario_name: {
                                    **scenario_data,
                                    'sys_messages': scenario_data['sys_messages'][:-1]
                                    # Remove the last (rejected) message
                                }
                                for scenario_name, scenario_data in role_data['scenarios'].items()
                                if not judged_messages[role_name]['scenarios'][scenario_name]['sys_messages'][-1][
                                    'acceptable']
                            }
                        }
                        for role_name, role_data in judged_messages.items()
                        if any(
                            not scenario_data['sys_messages'][-1]['acceptable']
                            for scenario_data in role_data['scenarios'].values()
                        )
                    }

                    # Update out_roles with valid messages
                    for role_name, role_data in judged_messages.items():
                        for scenario_name, scenario_data in role_data['scenarios'].items():
                            if scenario_data['sys_messages'][-1]['acceptable']:
                                # Update only the sys_messages field in out_roles
                                out_roles[role_name]['scenarios'][scenario_name]['sys_messages'].append(
                                    scenario_data['sys_messages'][-1]
                                )


            except Exception as e:
                self.logger.error(f"Error processing category '{category_name}' at level {msg_level_index}: {e}")
                self.logger.error(traceback.format_exc())

        return out_roles

    def generate_and_judge_scenarios(self, input_roles: dict, logging=True):

        """
        Process categories in parallel and append results into a single output.
        Each category's sys_messages are added to scenarios[x]['sys_messages']['category_name'].
        """
        # Ensure scenarios are in dict format
        for role_v in input_roles.values():
            assert isinstance(role_v['scenarios'], dict)

        roles_with_scenarios = deepcopy(input_roles)

        def process_category_thread(category_name, category_dict):
            return category_name, self.process_category(category_name, category_dict, roles_with_scenarios)

        with ThreadPoolExecutor(max_workers=len(self.pressure_categories)) as executor:
            future_to_category = {
                executor.submit(process_category_thread, category_name, category_dict): category_name
                for category_name, category_dict in list(self.pressure_categories.items()) if category_name == 'time'
            }

            for future in tqdm(as_completed(future_to_category), total=len(future_to_category)):
                category_name = future_to_category[future]
                try:
                    category_name, category_results = future.result()
                    # Append sys_messages for each scenario into the corresponding category_name bucket
                    for role_name, role_data in category_results.items():
                        for scenario_name, scenario_data in role_data['scenarios'].items():
                            if 'sys_messages' not in roles_with_scenarios[role_name]['scenarios'][scenario_name]:
                                roles_with_scenarios[role_name]['scenarios'][scenario_name]['sys_messages'] = {}
                            roles_with_scenarios[role_name]['scenarios'][scenario_name]['sys_messages'][category_name] = \
                                scenario_data['sys_messages']
                except Exception as e:
                    if logging:
                        self.logger.error(f"Error processing category '{category_name}': {e}")
                        self.logger.error(traceback.format_exc())

        return roles_with_scenarios
