import json
import logging
from copy import deepcopy

from agents.api_conf import APIConfiguration
from .base import BasePipeline
from .modules.scenario_utils_funcs import ScenarioManager as FuncsScenarioManager
from .modules.scenario_utils_messages import ScenarioManager as MessagesScenarioManager
from .modules.scenario_utils_policies import ScenarioManager as PoliciesScenarioManager
from .modules.scenario_utils_states import ScenarioManager as StatesScenarioManager
from .modules.utils import save_to_disk


class PipelineScenarios(BasePipeline):
    """Main system that orchestrates all components"""

    def __init__(self, cfg, logger: logging.Logger, workspace_name: str,
                 workspace_desc: str,
                 workspace_alternative_forms: list,
                 domain_name: str, domain_desc: str, domain_alternative_forms: list):
        super().__init__(cfg)

        self.cfg = cfg
        self.logger = logger
        self.workspace = workspace_name
        self.workspace_desc = workspace_desc
        self.workspace_alternative_forms = workspace_alternative_forms
        self.domain = domain_name
        self.domain_desc = domain_desc
        self.domain_alternative_forms = domain_alternative_forms

        self.min_initial_scenarios_per_role = cfg.min_initial_scenarios_per_role
        self.min_chosen_scenarios_per_role = cfg.min_chosen_scenarios_per_role

        # self.n_retries = cfg.n_retries

        assert self.logger is not None

        api_conf = APIConfiguration(
            model_name=cfg.model.model_name,
            model_provider=cfg.model.model_provider,
            api_base=cfg.model.api_base,
            use_cache=cfg.model.use_cache,
        )

        self.states_scenario_manager = StatesScenarioManager(api_conf=api_conf, logger=logger,
                                                             workspace_name=self.workspace,
                                                             workspace_alternative_forms=self.workspace_alternative_forms,
                                                             workspace_desc=workspace_desc,
                                                             domain_name=self.domain,
                                                             domain_desc=self.domain_desc,
                                                             domain_alternative_forms=self.domain_alternative_forms,
                                                             output_schemas_conf=cfg.output_schemas,
                                                             prompts_conf=cfg.prompts,
                                                             object_storage_conf=cfg.object_storage,
                                                             temperature=cfg.model.temperature,
                                                             min_initial_scenarios_per_role=cfg.min_initial_scenarios_per_role,
                                                             min_chosen_scenarios_per_role=cfg.min_chosen_scenarios_per_role,
                                                             generation_batch_size=cfg.scenario_gen_batch_size)

        self.funcs_scenario_manager = FuncsScenarioManager(api_conf=api_conf, logger=logger,
                                                           workspace_name=self.workspace,
                                                           workspace_alternative_forms=self.workspace_alternative_forms,
                                                           workspace_desc=workspace_desc,
                                                           domain_name=self.domain,
                                                           domain_desc=self.domain_desc,
                                                           domain_alternative_forms=self.domain_alternative_forms,
                                                           output_schemas_conf=cfg.output_schemas,
                                                           prompts_conf=cfg.prompts,
                                                           object_storage_conf=cfg.object_storage,
                                                           temperature=cfg.model.temperature,
                                                           generation_batch_size=cfg.scenario_gen_batch_size)

        self.policies_scenario_manager = PoliciesScenarioManager(api_conf=api_conf, logger=logger,
                                                                 workspace_name=self.workspace,
                                                                 workspace_alternative_forms=self.workspace_alternative_forms,
                                                                 workspace_desc=workspace_desc,
                                                                 domain_name=self.domain,
                                                                 domain_desc=self.domain_desc,
                                                                 domain_alternative_forms=self.domain_alternative_forms,
                                                                 output_schemas_conf=cfg.output_schemas,
                                                                 prompts_conf=cfg.prompts,
                                                                 object_storage_conf=cfg.object_storage,
                                                                 temperature=cfg.model.temperature,
                                                                 generation_batch_size=cfg.scenario_gen_batch_size)

        self.messages_scenario_manager = MessagesScenarioManager(api_conf=api_conf, logger=logger,
                                                                 workspace_name=self.workspace,
                                                                 workspace_alternative_forms=self.workspace_alternative_forms,
                                                                 workspace_desc=workspace_desc,
                                                                 domain_name=self.domain,
                                                                 domain_desc=self.domain_desc,
                                                                 domain_alternative_forms=self.domain_alternative_forms,
                                                                 output_schemas_conf=cfg.output_schemas,
                                                                 prompts_conf=cfg.prompts,
                                                                 object_storage_conf=cfg.object_storage,
                                                                 temperature=cfg.model.temperature,
                                                                 generation_batch_size=cfg.scenario_gen_batch_size)

    def update_scenarios(self, prev_roles_with_scenarios: dict, new_roles_with_scenarios: dict) -> dict:
        out = deepcopy(prev_roles_with_scenarios)
        for role_k, role_v in new_roles_with_scenarios.items():
            if role_k not in prev_roles_with_scenarios:
                out[role_k] = role_v
            else:
                for scenario in role_v['scenarios'].values():
                    scenario_name = scenario['name']
                    out[role_k]['scenarios'][scenario_name].update(scenario)
        return out

    def remove_existing_scenario_judgements(self, roles: dict, judgment_fields: list) -> dict:
        out_workspace = deepcopy(roles[self.domain][self.workspace])
        for role_k in out_workspace:
            for scenario_k, scenario_v in out_workspace[role_k]['scenarios'].items():
                for field in judgment_fields:
                    if field in scenario_v:
                        out_workspace[role_k]['scenarios'][scenario_k].pop(field)
                    else:
                        self.logger.debug(
                            f"Field {field} not found in scenario {scenario_k} of role {role_k}. Skipping...")
        roles[self.domain][self.workspace] = out_workspace
        return roles

    def run(self, roles: dict, grounding_attack_vectors, grounding_n_samples, force_overwrite=False):
        curr_roles_with_scenarios = {}
        try:
            with open(self.cfg.object_storage.scenarios_states, 'r') as f:
                curr_roles_with_scenarios = json.load(f)
        except FileNotFoundError:
            self.logger.info("No existing scenarios file found. Creating new one.")

        # Check if we need to generate scenarios for this domain/workspace
        should_generate = force_overwrite
        if not force_overwrite:
            if (self.domain not in curr_roles_with_scenarios or
                    self.workspace not in curr_roles_with_scenarios.get(self.domain, {})):
                should_generate = True
                self.logger.info(
                    f"No existing scenarios for domain {self.domain} and workspace {self.workspace}. Generating new ones.")

        if should_generate:
            new_roles_with_scenarios = self.states_scenario_manager.generate_and_judge_scenarios(
                input_roles=roles,
                grounding_attack_vectors=grounding_attack_vectors,
                grounding_n_samples=grounding_n_samples,
                logging=True
            )
            self.logger.debug(f'New scenarios: {new_roles_with_scenarios}')

            # Initialize domain if not exists
            if self.domain not in curr_roles_with_scenarios:
                curr_roles_with_scenarios[self.domain] = {}

            # Update workspace data
            if self.workspace not in curr_roles_with_scenarios[self.domain]:
                curr_roles_with_scenarios[self.domain][self.workspace] = new_roles_with_scenarios
            else:
                # Update existing scenarios while preserving others
                curr_roles_with_scenarios[self.domain][self.workspace] = self.update_scenarios(
                    curr_roles_with_scenarios[self.domain][self.workspace],
                    new_roles_with_scenarios
                )

            save_to_disk(curr_roles_with_scenarios, self.cfg.object_storage.scenarios_states)

        # Run the functions scenarios
        # curr_roles_with_scenarios = self.remove_existing_scenario_judgements(curr_roles_with_scenarios,
        #                                                                      judgment_fields=['acceptable', 'feedback'])
        # funcs_scenarios = {}
        #
        # force_overwrite = True
        # if not force_overwrite:
        #     try:
        #         with open(self.cfg.object_storage.scenarios_funcs, 'r') as f:
        #             funcs_scenarios = json.load(f)
        #             curr_roles_with_scenarios = funcs_scenarios
        #     except FileNotFoundError as e:
        #         self.logger.error(f"Could not find scenarios_funcs file: {e}")
        # if not funcs_scenarios:
        #     print("Running func-scenarios-gen...")
        #     roles = curr_roles_with_scenarios[self.domain][self.workspace]
        #     new_roles_with_scenarios = self.funcs_scenario_manager.generate_and_judge_scenarios(input_roles=roles,
        #                                                                                         logging=True)
        #     self.logger.debug(f'New scenarios: {new_roles_with_scenarios}')
        #     curr_roles_with_scenarios[self.domain][self.workspace] = self.update_scenarios(
        #         curr_roles_with_scenarios[self.domain][self.workspace], new_roles_with_scenarios)
        #     save_to_disk(curr_roles_with_scenarios, self.cfg.object_storage.scenarios_funcs)
        #
        # force_overwrite = False
        # # Run the policies scenarios
        # curr_roles_with_scenarios = self.remove_existing_scenario_judgements(curr_roles_with_scenarios,
        #                                                                      judgment_fields=['acceptable', 'feedback'])
        # policies_scenarios = {}
        # if not force_overwrite:
        #     try:
        #         with open(self.cfg.object_storage.scenarios_policies, 'r') as f:
        #             policies_scenarios = json.load(f)
        #             curr_roles_with_scenarios = policies_scenarios
        #     except FileNotFoundError as e:
        #         self.logger.error(f"Could not find scenarios_policies file: {e}")
        # if not policies_scenarios:
        #     print("Running policies-scenarios-gen...")
        #     roles = curr_roles_with_scenarios[self.domain][self.workspace]
        #     new_roles_with_scenarios = self.policies_scenario_manager.generate_and_judge_scenarios(input_roles=roles,
        #                                                                                            logging=True)
        #     self.logger.debug(f'New scenarios: {new_roles_with_scenarios}')
        #     curr_roles_with_scenarios[self.domain][self.workspace] = self.update_scenarios(
        #         curr_roles_with_scenarios[self.domain][self.workspace], new_roles_with_scenarios)
        #     save_to_disk(curr_roles_with_scenarios, self.cfg.object_storage.scenarios_policies)
        #
        # # Run the messages scenarios
        # curr_roles_with_scenarios = self.remove_existing_scenario_judgements(curr_roles_with_scenarios,
        #                                                                      judgment_fields=['acceptable', 'feedback'])
        #
        # messages_scenarios = {}
        # if not force_overwrite:
        #     try:
        #         with open(self.cfg.object_storage.scenarios_messages, 'r') as f:
        #             messages_scenarios = json.load(f)
        #             curr_roles_with_scenarios = messages_scenarios
        #     except FileNotFoundError as e:
        #         self.logger.error(f"Could not find scenarios_messages file: {e}")
        # if not messages_scenarios:
        #     print("Running _messages-scenarios-gen...")
        #     roles = curr_roles_with_scenarios[self.domain][self.workspace]
        #     new_roles_with_scenarios = self.messages_scenario_manager.generate_and_judge_scenarios(
        #         input_roles=roles,
        #         logging=True)
        #     self.logger.debug(f'New scenarios: {new_roles_with_scenarios}')
        #     curr_roles_with_scenarios[self.domain][self.workspace] = self.update_scenarios(
        #         curr_roles_with_scenarios[self.domain][self.workspace], new_roles_with_scenarios)
        #     save_to_disk(curr_roles_with_scenarios, self.cfg.object_storage.scenarios_messages)
