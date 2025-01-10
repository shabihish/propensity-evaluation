import logging
import json
import re
from copy import deepcopy

from agents.api_conf import APIConfiguration
from sim.base import Simulator
from agents.agent import Agent

from utils.regex import remove_duplicate_names_by_prefix, apply_to_all_keys
from sim.modules.utils import read_prompts, load_output_schemas, save_to_disk, run_agent_query
from sim.modules.role_utils import validate_roles, RoleManager
from sim.modules.scenario_utils import ScenarioManager


class PipelineSimulator(Simulator):
    """Main system that orchestrates all components"""

    def __init__(self, cfg, logger: logging.Logger, workspace_name: str, workspace_alternative_forms: list,
                 domain_name: str, domain_alternative_forms: list):
        super().__init__(cfg)

        self.cfg = cfg
        self.logger = logger
        self.workspace = workspace_name
        self.workspace_alternative_forms = workspace_alternative_forms
        self.domain = domain_name
        self.domain_alternative_forms = domain_alternative_forms

        self.n_initial_roles = cfg.n_initial_roles
        self.n_min_initial_roles = cfg.n_min_initial_roles
        self.n_roles = cfg.n_roles
        self.n_scenarios_per_role = cfg.n_scenarios_per_role

        self.n_retries = cfg.n_retries

        assert self.logger is not None

        api_conf = APIConfiguration(
            model_name=cfg.model.model_name,
            model_provider=cfg.model.model_provider,
            api_base=cfg.model.api_base,
            use_cache=cfg.model.use_cache,
        )

        self.roles_manager = RoleManager(api_conf=api_conf, logger=logger,
                                         workspace_name=self.workspace,
                                         workspace_alternative_forms=self.workspace_alternative_forms,
                                         domain_name=self.domain,
                                         domain_alternative_forms=self.domain_alternative_forms,
                                         output_schemas_conf=cfg.output_schemas,
                                         prompts_conf=cfg.prompts, object_storage_conf=cfg.object_storage,
                                         n_initial_roles=self.n_initial_roles, temperature=cfg.model.temperature)

        self.scenario_manager = ScenarioManager(api_conf=api_conf, logger=logger,
                                                workspace_name=self.workspace,
                                                workspace_alternative_forms=self.workspace_alternative_forms,
                                                domain_name=self.domain,
                                                domain_alternative_forms=self.domain_alternative_forms,
                                                output_schemas_conf=cfg.output_schemas,
                                                prompts_conf=cfg.prompts, object_storage_conf=cfg.object_storage,
                                                temperature=cfg.model.temperature,
                                                generation_batch_size=cfg.scenario_gen_batch_size)

        self.actor_manager = ActorManager(api_conf=api_conf, logger=logger,
                                          workspace_name=self.workspace,
                                          workspace_alternative_forms=self.workspace_alternative_forms,
                                          domain_name=self.domain,
                                          domain_alternative_forms=self.domain_alternative_forms,
                                          output_schemas_conf=cfg.output_schemas,
                                          prompts_conf=cfg.prompts, object_storage_conf=cfg.object_storage,
                                          temperature=cfg.model.temperature, )

        # self.actors_generation_agent = Agent(
        #     sys_prompt=load_sys_prompt(cfg.sys_prompts.actors_generation_agent),
        #     api_conf=api_conf,
        #     output_schema=load_output_schemas(
        #         cfg.output_schemas.actors_generation_agent),
        #     temperature=cfg.model.temperature)
        # self.scenarios_revision_agent = Agent(
        #     sys_prompt=load_sys_prompt(cfg.sys_prompts.scenarios_revision_agent),
        #     api_conf=api_conf,
        #     output_schema=load_output_schemas(
        #         cfg.output_schemas.scenarios_revision_agent),
        #     temperature=cfg.model.temperature)
        #
        # self.actors_generation_agent = Agent(sys_prompt=load_sys_prompt(cfg.sys_prompts.actors_generation_agent),
        #                                      api_conf=api_conf,
        #                                      output_schema=load_output_schemas(
        #                                          cfg.output_schemas.actors_generation_agent),
        #                                      temperature=cfg.model.temperature)
        # self.actors_revision_agent = Agent(sys_prompt=load_sys_prompt(cfg.sys_prompts.actors_revision_agent),
        #                                    api_conf=api_conf,
        #                                    output_schema=load_output_schemas(
        #                                        cfg.output_schemas.actors_revision_agent),
        #                                    temperature=cfg.model.temperature)
        #
        # self.use_cases_generation_agent = Agent(
        #     sys_prompt=load_sys_prompt(cfg.sys_prompts.use_cases_generation_agent),
        #     api_conf=api_conf,
        #     output_schema=load_output_schemas(
        #         cfg.output_schemas.use_cases_generation_agent),
        #     temperature=cfg.model.temperature)
        # self.use_cases_revision_agent = Agent(
        #     sys_prompt=load_sys_prompt(cfg.sys_prompts.use_cases_revision_agent),
        #     api_conf=api_conf,
        #     output_schema=load_output_schemas(
        #         cfg.output_schemas.use_cases_revision_agent),
        #     temperature=cfg.model.temperature)
        # self.class_diagram_generation_agent = Agent(
        #     sys_prompt=load_sys_prompt(cfg.sys_prompts.class_diagram_generation_agent),
        #     api_conf=api_conf,
        #     output_schema=load_output_schemas(
        #         cfg.output_schemas.class_diagram_generation_agent),
        #     temperature=cfg.model.temperature)
        # self.class_diagram_revision_agent = Agent(
        #     sys_prompt=load_sys_prompt(cfg.sys_prompts.class_diagram_revision_agent),
        #     api_conf=api_conf,
        #     output_schema=load_output_schemas(
        #         cfg.output_schemas.class_diagram_revision_agent),
        #     temperature=cfg.model.temperature)
        #
        # self.seq_diagram_generation_agent = Agent(
        #     sys_prompt=load_sys_prompt(cfg.sys_prompts.seq_diagram_generation_agent),
        #     api_conf=api_conf,
        #     output_schema=load_output_schemas(
        #         cfg.output_schemas.seq_diagram_generation_agent),
        #     temperature=cfg.model.temperature)
        # self.seq_diagram_revision_agent = Agent(
        #     sys_prompt=load_sys_prompt(cfg.sys_prompts.seq_diagram_revision_agent),
        #     api_conf=api_conf,
        #     output_schema=load_output_schemas(
        #         cfg.output_schemas.seq_diagram_revision_agent),
        #     temperature=cfg.model.temperature)

    def check_for_missing_fields(self, input_dict: dict, required_fields: list) -> dict:
        out = {}
        for obj in input_dict.values():
            missing_fields = []
            for required_field in required_fields:
                if required_field not in obj:
                    missing_fields.append(required_field)
            if missing_fields:
                out[obj['name']] = {'name': obj['name'], 'missing_fields': missing_fields}
        return out

    def generate_roles(self, n_roles, initial_role_names: list = None) -> dict:
        all_roles = {}
        prev_roles = [] if initial_role_names is None else initial_role_names

        if n_roles <= 0:
            return all_roles
        while len(all_roles) < n_roles:
            prompt = read_prompts(self.cfg.prompts.roles_agents, key='USER_GEN',
                                  context={**vars(self), 'prev_roles': prev_roles}, logger=self.logger)
            try:
                response = run_agent_query(self.n_retries, self.logger, prompt, self.roles_generation_agent,
                                           to_json=True,
                                           json_transform_keys=['roles', 'name'])
                self.validate_roles(response, required_fields=['name', 'description'], n_roles=0)
                all_roles.update(response)
                prev_roles = list(all_roles.keys())
            except Exception as e:
                self.logger.error(f"Error in generate_roles: {e}")
                continue

        return all_roles

    # def revise_roles(self, roles_with_feedback: dict) -> dict:
    #     roles_to_revise = {name: role for name, role in roles_with_feedback.items() if not role['is_accepted']}
    #     # Remove the is_accepted flag from roles to revise
    #     for role in roles_to_revise.values():
    #         role.pop('is_accepted', None)
    #
    #     revised_roles = set()
    #     while roles_to_revise:
    #         prompt = read_prompts(self.cfg.prompts.roles_agents, key='USER_REV',
    #                               context={'roles_with_feedback': roles_to_revise},
    #                               logger=self.logger)
    #
    #         try:
    #             response = self.run_agent_query(prompt, self.roles_revision_agent, to_json=True,
    #                                             json_transform_keys=['roles', 'name'])
    #             self.validate_roles(response, required_fields=['name', 'description'], n_roles=0)
    #         except Exception as e:
    #             self.logger.error(f"Error in revise_roles: {e}")
    #             continue
    #
    #         # Add revised roles to the output
    #         for name, revised_role in response.items():
    #             roles_with_feedback[name] = revised_role
    #             revised_roles.add(name)
    #
    #
    #         # Remove revised roles from the list to revise
    #         roles_to_revise = {name: role for name, role in roles_to_revise.items() if name not in revised_roles}
    #
    #     return roles_with_feedback

    # def revise_roles(self, roles_with_feedback: dict) -> dict:
    #     for name, role in roles_with_feedback.items():
    #         if not role['is_accepted']:
    #             # Remove the is_accepted flag from the role
    #             role.pop('is_accepted', None)
    #
    #             # Prepare the prompt for the revise agent
    #             prompt = read_prompts(self.cfg.prompts.roles_agents, key='USER_REV',
    #                                   context={'roles_with_feedback': {name: role}},
    #                                   logger=self.logger)
    #
    #             while True:
    #                 try:
    #                     response = self.run_agent_query(prompt, self.roles_revision_agent, to_json=True,
    #                                                     json_transform_keys=['roles', 'name'])
    #                     self.validate_roles(response, required_fields=['name', 'description'], n_roles=0)
    #                     break
    #                 except Exception as e:
    #                     self.logger.error(f"Error in revise_roles for role {name}: {e}")
    #                     continue
    #
    #             # Update the role in the original set of roles
    #             roles_with_feedback.pop(name)
    #             new_name = list(response.keys())[0]
    #             roles_with_feedback[new_name] = response[new_name]
    #
    #     return roles_with_feedback

    def revise_roles(self, roles_with_feedback: dict) -> dict:
        roles_to_revise = {name: role for name, role in roles_with_feedback.items() if not role['is_accepted']}
        for role in roles_to_revise.values():
            role.pop('is_accepted', None)

        revised_roles = set()
        retries = 0
        while roles_to_revise:
            self.logger.debug(f"Roles to revise: {list(roles_to_revise.keys())}")
            prompt = read_prompts(self.cfg.prompts.roles_agents, key='USER_REV',
                                  context={'roles_with_feedback': roles_to_revise},
                                  logger=self.logger)

            try:
                response = run_agent_query(self.n_retries, self.logger, prompt, self.roles_revision_agent, to_json=True,
                                           json_transform_keys=['roles', 'name'])
                self.validate_roles(response, required_fields=['name', 'description', 'change_log'], n_roles=0)
            except Exception as e:
                self.logger.error(f"Error in revise_roles: {e}")
                retries += 1
                if retries > 5 and len(roles_with_feedback) > self.n_min_initial_roles:
                    return roles_with_feedback
                continue

            self.logger.debug(f"Reviser response: {response}")
            for name, revised_role in response.items():
                replaces = revised_role.get('replaces_role_name', None)
                revised = False
                if name in roles_with_feedback:
                    roles_with_feedback[name] = revised_role
                    revised = True
                elif replaces is not None and replaces in roles_with_feedback:
                    roles_with_feedback.pop(replaces)
                    roles_with_feedback[name] = revised_role
                    revised = True

                if revised:
                    revised_roles.add(name)
                    self.logger.debug(f"Revised role: {name}")

            roles_to_revise = {name: role for name, role in roles_to_revise.items() if name not in revised_roles}

        return roles_with_feedback

    # def generate_scenarios(self, roles: dict) -> dict:
    #     pass
    #
    # def revise_scenarios(self, scenarios: dict) -> dict:
    #     pass
    #
    # def generate_actors(self, scenarios: dict) -> dict:
    #     pass
    #
    # def revise_actors(self, actors: dict) -> dict:
    #     pass
    #
    # def generate_use_cases(self, actors: dict) -> dict:
    #     pass
    #
    # def revise_use_cases(self, use_cases: dict) -> dict:
    #     pass
    #
    # def generate_class_diagram(self, use_cases: dict) -> dict:
    #     pass
    #
    # def revise_class_diagram(self, class_diagram: dict) -> dict:
    #     pass
    #
    # def generate_seq_diagram(self, class_diagram: dict) -> dict:
    #     pass
    #
    # def revise_seq_diagram(self, seq_diagram: dict) -> dict:
    #     pass

    # def judge_roles(self, input_scenarios: dict) -> tuple:
    #     """
    #     Validate the roles generated by the roles generation agent
    #     :param input_scenarios: a dictionary of the generated roles
    #     :param required_fields: a list of required fields in each role
    #     :return: is_accepted, feedback
    #     """
    #     # Run the judge model
    #     prompt = read_prompts(self.cfg.prompts.judge_agents, key='USER_ROLES_VERIF',
    #                           context={'roles': str(input_scenarios)},
    #                           logger=self.logger)
    #     passes_requirements = False
    #     while not passes_requirements:
    #         passes_requirements = True
    #         # Ask the judge model for feedback
    #         response = self.run_agent_query(prompt, self.roles_verif_judge, to_json=True,
    #                                         json_transform_keys=['roles', 'name'])
    #
    #         is_accepted = True
    #         out = deepcopy(input_scenarios)
    #         try:
    #             for role in out.values():
    #                 self.logger.debug(f"Checking feedback for role: {role['name']}")
    #                 if role['name'] not in response:
    #                     raise ValueError(f"Role {role['name']} not found in response")
    #                 role['feedback'] = response[role['name']]['feedback']
    #                 role['is_accepted'] = response[role['name']]['is_accepted']
    #                 if not role['is_accepted']:
    #                     is_accepted = False
    #         except Exception as e:
    #             passes_requirements = False
    #             self.logger.error(f"Error in judge_roles: {e}")
    #     return is_accepted, out

    # def judge_roles(self, input_scenarios: dict) -> tuple:
    #     """
    #     Validate the roles generated by the roles generation agent
    #     :param input_scenarios: a dictionary of the generated roles
    #     :return: is_accepted, feedback
    #     """
    #     missing_roles = list(input_scenarios.keys())
    #     out = deepcopy(input_scenarios)
    #     passes_requirements = False
    #
    #     while not passes_requirements:
    #         if missing_roles:
    #             self.logger.debug(f"Missing roles: {missing_roles}")
    #         prompt = read_prompts(self.cfg.prompts.judge_agents, key='USER_ROLES_VERIF',
    #                               context={'roles': str({name: input_scenarios[name] for name in missing_roles})},
    #                               logger=self.logger)
    #         passes_requirements = True
    #         response = run_agent_query(self.n_retries, self.logger, prompt, self.roles_verif_judge, to_json=True,
    #                                    json_transform_keys=['roles', 'name'])
    #
    #         new_missing_roles = []
    #         for role_name in missing_roles:
    #             try:
    #                 self.logger.debug(f"Checking feedback for role: {role_name}")
    #                 if role_name not in response:
    #                     new_missing_roles.append(role_name)
    #                     passes_requirements = False
    #                     continue
    #                 out[role_name]['feedback'] = response[role_name]['feedback']
    #                 out[role_name]['is_accepted'] = response[role_name]['is_accepted']
    #             except Exception as e:
    #                 new_missing_roles.append(role_name)
    #                 passes_requirements = False
    #                 self.logger.error(f"Error in judge_roles: {e}")
    #
    #         missing_roles = new_missing_roles
    #
    #     is_accepted = all(role['is_accepted'] for role in out.values())
    #     return is_accepted, out
    #
    # def generate_and_judge_initial_roles(self, logging=True) -> dict:
    #     try:
    #         with open(self.cfg.finalized_jsons.roles, 'r') as f:
    #             curr_accepted_roles = json.load(f)
    #         # if len(curr_accepted_roles) >= self.n_initial_roles:
    #         #     return curr_accepted_roles
    #     except FileNotFoundError as e:
    #         self.logger.error(f"Error in generate_and_judge_initial_roles: {e}")
    #         curr_accepted_roles = {}
    #
    #     accepted_roles = curr_accepted_roles
    #     while True:
    #         initial_role_names = list(curr_accepted_roles.keys()) if curr_accepted_roles else None
    #         n_roles_to_generate = max(0, self.n_initial_roles - len(accepted_roles))
    #         if logging:
    #             self.logger.debug(f'Roles left to generate: {n_roles_to_generate}')
    #
    #         # Generate roles
    #         generated_roles = self.generate_roles(n_roles=n_roles_to_generate, initial_role_names=initial_role_names)
    #         if logging:
    #             self.logger.debug(f'Generated roles_dict: {generated_roles}\n\n')
    #
    #         is_accepted, judged_roles = self.judge_roles(generated_roles)
    #         curr_accepted_roles = {name: role for name, role in judged_roles.items() if role['is_accepted']}
    #         if logging:
    #             self.logger.debug(f'Output for judge_roles: {judged_roles}, {is_accepted}')
    #
    #         curr_accepted_roles = remove_duplicate_names_by_prefix(curr_accepted_roles, rf'{self.workspace}\s*', 'name',
    #                                                                flags=re.IGNORECASE)
    #         curr_accepted_roles = remove_duplicate_names_by_prefix(curr_accepted_roles, rf'{self.domain}\s*', 'name',
    #                                                                flags=re.IGNORECASE)
    #         curr_accepted_roles = apply_to_all_keys(curr_accepted_roles, func=str.title, key_to_update='name')
    #         curr_accepted_roles = apply_to_all_keys(curr_accepted_roles, func=lambda x: re.sub(r'[-_]', ' ', x),
    #                                                 key_to_update='name')
    #
    #         accepted_roles.update(curr_accepted_roles)
    #         if logging:
    #             self.logger.debug(f'Accepted role names: {list(accepted_roles.keys())}\n\n')
    #
    #         if len(accepted_roles) >= self.n_initial_roles:
    #             break
    #     save_to_disk(accepted_roles, self.cfg.finalized_jsons.roles)
    #     return accepted_roles

    def run(self) -> None:
        initial_roles = self.roles_manager.generate_and_judge_initial_roles(logging=True)
        for role in initial_roles.values():
            role.pop('is_accepted', None)
            role.pop('feedback', None)
        self.logger.debug(f'Initial roles: {initial_roles}')

        # initial_roles = {k: v for i, (k, v) in enumerate(initial_roles.items()) if i < 4}
        scenarios = self.scenario_manager.generate_and_judge_scenarios(input_roles=initial_roles, logging=True)
        self.logger.debug(f'Initial scenarios: {scenarios}')
