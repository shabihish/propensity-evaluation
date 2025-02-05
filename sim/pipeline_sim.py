import logging
import json
from agents.api_conf import APIConfiguration
from sim.base import Simulator
from sim.modules.role_utils import RoleManager
from sim.modules.scenario_utils_states import ScenarioManager as statesScenarioManager
from sim.modules.scenario_utils_policies import ScenarioManager as PoliciesScenarioManager
from sim.modules.scenario_utils_states import ScenarioManager as statesScenarioManager
from sim.modules.utils import read_prompts, run_agent_query, save_to_disk


class PipelineSimulator(Simulator):
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

        self.n_initial_roles = cfg.n_initial_roles
        self.n_min_initial_roles = cfg.n_min_initial_roles
        self.n_roles = cfg.n_roles
        # self.n_scenarios_per_role = cfg.n_scenarios_per_role

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
                                         workspace_desc=workspace_desc,
                                         domain_name=self.domain,
                                         domain_alternative_forms=self.domain_alternative_forms,
                                         output_schemas_conf=cfg.output_schemas,
                                         prompts_conf=cfg.prompts, object_storage_conf=cfg.object_storage,
                                         n_initial_roles=self.n_initial_roles, temperature=cfg.model.temperature)

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
                                                             generation_batch_size=cfg.scenario_gen_batch_size)


    def run(self) -> None:
        # load initial roles
        try:
            with open(self.cfg.object_storage.roles, 'r') as f:
                initial_roles = json.load(f)
                # all_roles = self.normalize_roles(all_roles, remove_similar=False)
        except FileNotFoundError as e:
            self.logger.error(f"Error in generate_and_judge_initial_roles: {e}")
            initial_roles = {}

        if not initial_roles.get(self.domain, {}).get(self.workspace, {}):
            new_roles = self.roles_manager.generate_and_judge_initial_roles(logging=True)
            initial_roles.setdefault(self.domain, {})[self.workspace] = new_roles
            save_to_disk(initial_roles, self.cfg.object_storage.roles)

        for domain in initial_roles:
            for workspace in initial_roles[domain]:
                for role in initial_roles[domain][workspace].values():
                    role.pop('is_accepted', None)
                    role.pop('feedback', None)
                self.logger.debug(f'Initial roles: {initial_roles}')

        initial_roles = {k: v for i, (k, v) in enumerate(initial_roles[self.domain][self.workspace].items()) if i < 4}
        scenarios = self.states_scenario_manager.generate_and_judge_scenarios(input_roles=initial_roles, logging=True)
        self.logger.debug(f'Initial scenarios: {scenarios}')
