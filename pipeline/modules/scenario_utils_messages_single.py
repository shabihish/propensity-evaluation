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
from .utils import json_obj_list_to_dict
import logging
from tqdm import tqdm
from pydantic import BaseModel


class Message(BaseModel):
    from_: str  # Use `from_` since `from` is a reserved keyword in Python
    subject: str
    body: str


class ScenariosGenMessagesSingle(BaseModel):
    message: Message


class JudgeSingleMessages(BaseModel):
    acceptable: bool
    feedback: str


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
        self.pressure_levels = [
            # {"Time.A": 2, "A": 2, "B": 2, "C": 2, "D": 2, "E": 2},
            # {"Time.A": 4, "A": 4, "B": 4, "C": 4, "D": 4, "E": 4},
            # {"Time.A": 5, "A": 5, "B": 5, "C": 5, "D": 5, "E": 5},
            {"Time.A": 6, "A": 6, "B": 6, "C": 6, "D": 6, "E": 6},
            {"Time.A": 7, "A": 7, "B": 7, "C": 7, "D": 7, "E": 7},
            {"Time.A": 8, "A": 8, "B": 8, "C": 8, "D": 8, "E": 8},
            {"Time.A": 9, "A": 9, "B": 9, "C": 9, "D": 9, "E": 9},
            {"Time.A": 10, "A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
            {"Time.A": 10, "A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
            # {"Time.A": 10, "A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
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
        # output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_messages_single)
        output_schema = ScenariosGenMessagesSingle
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

        # output_schema = load_output_schemas(self.output_schemas_conf.judge_single_messages)
        output_schema = JudgeSingleMessages
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature
        )

    def generate_new_message(self, category_name: str, msg_level_index: int, role_name: str, scenario_dict: dict,
                             previous_sys_messages: list, curr_msg_requirements: str, curr_authorities_list: str):
        # Initialize the agent for the current category and level
        curr_agent = self.build_generation_agent(
            pressure_category=category_name,
            message_requirements=curr_msg_requirements,
            authorities_list=curr_authorities_list,
        )

        prompt = read_prompts(self.prompts_conf.scenarios_agents_messages_single, key='USER_GEN',
                              context={
                                  'role_name': role_name,
                                  'scenario': scenario_dict,
                                  'previous_sys_messages': json.dumps(previous_sys_messages,
                                                                      indent=2) if previous_sys_messages else 'None',
                              },
                              logger=self.logger)
        # Generate new sys-message
        new_message = None
        while not new_message:
            try:
                response = run_agent_query(
                    prompt=prompt,
                    agent=curr_agent,
                    logger=self.logger,
                    to_json=True
                )
                response = response['message']
                # Validate and store the generated message
                assert 'from_' in response, "Missing 'from' field in response"
                assert 'subject' in response, "Missing 'subject' field in response"
                assert 'body' in response, "Missing 'body' field in response"
                response['from'] = response['from_']
                del response['from_']

                new_message = response
                print(f"New message: {new_message}")
            except Exception as e:
                self.logger.error(
                    f"Error generating sys-message for level {category_name}{msg_level_index + 1} for role '{role_name}' in scenario '{scenario_dict['name']}': {e}")
                self.logger.error(traceback.format_exc())

        return new_message

    def judge_message(self, category_name: str, msg_level_index: int, role_name: str, scenario_dict: dict,
                      previous_sys_messages: list, curr_msg_requirements: str, curr_authorities_list: str,
                      new_message: dict):

        # Initialize the agent for the current category and level
        curr_agent = self.build_judgement_agent(
            pressure_category=category_name,
            message_requirements=curr_msg_requirements,
            authorities_list=curr_authorities_list,
        )
        prompt = read_prompts(self.prompts_conf.judge_agents, key='USER_SINGLE_MESSAGES_VERIF',
                              context={
                                  'role_name': role_name,
                                  'scenario': scenario_dict,
                                  'previous_sys_messages': json.dumps(previous_sys_messages,
                                                                      indent=2) if previous_sys_messages else 'None',
                                  'new_sys_message': json.dumps(new_message, indent=2)
                              },
                              logger=self.logger)

        # Run judge
        judgement = None
        while not judgement:
            try:
                response = run_agent_query(
                    prompt=prompt,
                    agent=curr_agent,
                    logger=self.logger,
                    to_json=True
                )

                # Validate and store the generated message
                assert 'acceptable' in response, "Missing 'acceptable' field in judgement"
                assert 'feedback' in response, "Missing 'feedback' field in judgement"

                judgement = response
            except Exception as e:
                self.logger.error(
                    f"Error judging sys-message for level {category_name}{msg_level_index + 1} for role '{role_name}' in scenario '{scenario_dict['name']}': {e}")
                self.logger.error(traceback.format_exc())

        return judgement

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

    def generate_scenario_messages(self, role_name: str, scenario_data: dict):
        all_messages = {}
        for category_name, category_dict in self.pressure_categories.items():
            # if category_name != 'time':
            #     continue
            previous_sys_messages = []
            for msg_level_index in range(len(self.pressure_levels)):
                try:
                    # print(category_name, msg_level_index)
                    curr_level = self.pressure_levels[msg_level_index]

                    # Build message requirements and authorities list
                    curr_msg_requirements = self.build_message_requirements(category_dict, curr_level)
                    curr_authorities_list = self.authorities_levels[msg_level_index]

                    # print("curr_msg_requirements:\n", curr_msg_requirements)
                    # print("curr_authorities_list:\n", curr_authorities_list)

                    # Generate new message for the current category and level
                    new_message = self.generate_new_message(
                        category_name=category_name,
                        msg_level_index=msg_level_index,
                        role_name=role_name,
                        scenario_dict=scenario_data,
                        previous_sys_messages=previous_sys_messages,
                        curr_msg_requirements=curr_msg_requirements,
                        curr_authorities_list=curr_authorities_list
                    )

                    # Judge the generated message
                    judgement = self.judge_message(
                        category_name=category_name,
                        msg_level_index=msg_level_index,
                        role_name=role_name,
                        scenario_dict=scenario_data,
                        previous_sys_messages=previous_sys_messages,
                        curr_msg_requirements=curr_msg_requirements,
                        curr_authorities_list=curr_authorities_list,
                        new_message=new_message
                    )

                    if judgement['acceptable']:
                        # Add the generated message to the scenario_data
                        new_message['feedback'] = judgement['feedback']
                        # print(f"New message: {new_message}")
                        previous_sys_messages.append(new_message)
                    else:
                        # Re-process the level
                        msg_level_index -= 1
                except Exception as e:
                    self.logger.error(
                        f"From generate_scenario_messages: Error generating sys-message for level {category_name}{msg_level_index + 1} for role '{role_name}' in scenario '{scenario_data['name']}': {e}")
                    self.logger.error(traceback.format_exc())
            all_messages[category_name] = previous_sys_messages

        return all_messages

    def generate_and_judge_scenarios(self, input_roles: dict, logging=True):
        # Expects the scenarios to be converted to a dict already
        for role_v in input_roles.values():
            assert isinstance(role_v['scenarios'], dict)

        roles_with_scenarios = deepcopy(input_roles)

        # for role, role_data in input_roles.items():
        #     for scenario_name, scenario_data in role_data['scenarios'].items():
        #         all_messages = self.generate_scenario_messages(role, scenario_data)
        #         roles_with_scenarios[role]['scenarios'][scenario_name]['sys_messages'] = all_messages
        #         break
        #     break
        def process_role_scenario(role, scenario_name, scenario_data):
            all_messages = self.generate_scenario_messages(role, scenario_data)
            return role, scenario_name, all_messages

        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_role_scenario = {
                executor.submit(process_role_scenario, role, scenario_name, scenario_data): (role, scenario_name)
                for role, role_data in input_roles.items()
                for scenario_name, scenario_data in role_data['scenarios'].items()
            }

            for future in tqdm(as_completed(future_to_role_scenario)):
                role, scenario_name = future_to_role_scenario[future]
                try:
                    _, _, all_messages = future.result()
                    roles_with_scenarios[role]['scenarios'][scenario_name]['sys_messages'] = all_messages
                except Exception as e:
                    if logging:
                        self.logger.error(f"Error processing role '{role}', scenario '{scenario_name}': {e}")
        return roles_with_scenarios
