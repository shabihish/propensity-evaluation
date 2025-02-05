import logging
import json
from agents.api_conf import APIConfiguration
from sim.base import Simulator
from sim.modules.role_utils import RoleManager
from sim.modules.scenario_utils_states import ScenarioManager as statesScenarioManager
from sim.modules.scenario_utils_funcs import ScenarioManager as FuncsScenarioManager
from sim.modules.scenario_utils_policies import ScenarioManager as PoliciesScenarioManager
from sim.modules.scenario_utils_messages import ScenarioManager as MessagesScenarioManager
from sim.modules.utils import read_prompts, run_agent_query, save_to_disk


class ScenariosSim(Simulator):
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

        self.states_scenario_manager = statesScenarioManager(api_conf=api_conf, logger=logger,
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
                                                                 min_initial_scenarios_per_role=cfg.min_initial_scenarios_per_role,
                                                                 min_chosen_scenarios_per_role=cfg.min_chosen_scenarios_per_role,
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
                                                                 min_initial_scenarios_per_role=cfg.min_initial_scenarios_per_role,
                                                                 min_chosen_scenarios_per_role=cfg.min_chosen_scenarios_per_role,
                                                                 generation_batch_size=cfg.scenario_gen_batch_size)

    def update_scenarios(self, prev_roles_with_scenarios: dict, new_roles_with_scenarios: dict) -> dict:
        for role_k, role_v in new_roles_with_scenarios.items():
            if role_k not in prev_roles_with_scenarios:
                prev_roles_with_scenarios[role_k] = role_v
            else:
                for scenario in role_v['scenarios']:
                    prev_roles_with_scenarios[role_k]['scenarios']
        pass

    def run(self, roles: dict) -> None:
        curr_roles_with_scenarios = {}
        try:
            with open(self.cfg.object_storage.scenarios_states, 'r') as f:
                curr_roles_with_scenarios = json.load(f)
        except FileNotFoundError as e:
            self.logger.error(f"Error in generate_and_judge_initial_scenarios: {e}")

        if not curr_roles_with_scenarios:
            new_roles_with_scenarios = self.states_scenario_manager.generate_and_judge_scenarios(input_roles=roles,
                                                                                                 logging=True)
            self.logger.debug(f'New scenarios: {new_roles_with_scenarios}')

            if self.domain not in curr_roles_with_scenarios:
                curr_roles_with_scenarios[self.domain] = {}
            curr_roles_with_scenarios[self.domain][self.workspace] = new_roles_with_scenarios
            save_to_disk(curr_roles_with_scenarios, self.cfg.object_storage.scenarios_states)

        # # Run the functions scenarios
        print("Running func-scenarios-gen...")
        roles = curr_roles_with_scenarios[self.domain][self.workspace]
        new_roles_with_scenarios = self.funcs_scenario_manager.generate_and_judge_scenarios(input_roles=roles,
                                                                                            logging=True)
        self.logger.debug(f'New scenarios: {new_roles_with_scenarios}')
        curr_roles_with_scenarios[self.domain][self.workspace] = new_roles_with_scenarios
        save_to_disk(curr_roles_with_scenarios, self.cfg.object_storage.scenarios_funcs)
