import json
import re
from copy import deepcopy
from utils.regex import remove_duplicate_names_by_prefix, apply_to_all_keys
from agents.agent import Agent
from sim.modules.utils import read_prompts, save_to_disk, run_agent_query, check_for_missing_fields, load_output_schemas
from sim.modules.utils import capitalize_first_letter, normalize_string
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sim.modules.graph_utils import SimilarityGraph


def validate_roles(roles: dict, required_fields: list, n_roles: int):
    if len(roles) < n_roles:
        raise RuntimeError(f"Expected at least {n_roles} roles, got {len(roles)} roles.")

    failed_roles = check_for_missing_fields(roles, required_fields)
    if failed_roles:
        raise RuntimeError(f"Roles missing some or all of the required fields: {failed_roles}.")


class RoleManager:
    def __init__(self, api_conf, logger, workspace_name, workspace_alternative_forms: list, domain_name,
                 domain_alternative_forms: list, prompts_conf: DictConfig,
                 output_schemas_conf: DictConfig, object_storage_conf: DictConfig,
                 n_initial_roles, temperature):
        self.api_conf = api_conf
        self.logger = logger
        self.workspace = workspace_name
        self.workspace_alternative_forms = workspace_alternative_forms
        self.domain = domain_name
        self.domain_alternative_forms = domain_alternative_forms
        self.prompts_conf = prompts_conf
        self.output_schemas_conf = output_schemas_conf
        self.object_storage_conf = object_storage_conf
        self.n_initial_roles = n_initial_roles
        self.temperature = temperature

        self.roles_generation_agent = self._init_roles_generation_agent()
        # self.roles_revision_agent = self._init_roles_revision_agent()
        self.roles_verif_judge = self._init_roles_verif_judge()

        self.similarity_graph = SimilarityGraph(TfidfVectorizer, threshold=0.5)

    def _init_roles_generation_agent(self):
        # Roles generation agent
        sys_prompt = read_prompts(self.prompts_conf.roles_agents, key='SYS_GEN',
                                  context={'workspace': self.workspace, 'domain': self.domain,
                                           'n_initial_roles': self.n_initial_roles}, logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.roles_generation)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def _init_roles_revision_agent(self):
        # Roles revision agent
        sys_prompt = read_prompts(self.prompts_conf.roles_agents, key='SYS_REV',
                                  context={'workspace': self.workspace, 'domain': self.domain}, logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.roles_revision)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def _init_roles_verif_judge(self):
        sys_prompt = read_prompts(self.prompts_conf.judge_agents, key='SYS_ROLES_VERIF',
                                  context={'workspace': self.workspace, 'domain': self.domain},
                                  logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.judge_roles)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def normalize_roles(self, roles: dict, remove_similar=False) -> dict:
        # Update the keys in role dict
        roles = apply_to_all_keys(roles, func=normalize_string, key_to_update='name', lower=False)

        for w in [self.workspace] + self.workspace_alternative_forms:
            roles = remove_duplicate_names_by_prefix(roles, rf'{w}\s*', 'name', flags=re.IGNORECASE)
        for d in [self.domain] + self.domain_alternative_forms:
            roles = remove_duplicate_names_by_prefix(roles, rf'{d}\s*', 'name', flags=re.IGNORECASE)
        # roles = apply_to_all_keys(roles, func=lambda x: re.sub(r'[-_]', ' ', x), key_to_update='name')
        roles = apply_to_all_keys(roles, func=capitalize_first_letter, key_to_update='name')

        # normalize the textual fields in the roles
        for role in roles.values():
            for field in role.keys():
                if isinstance(role[field], str) and field != 'name':
                    role[field] = normalize_string(role[field], lower=True)

        if remove_similar:
            # Form the similarity graph and remove similar entries
            graph_data = {x: f'{x}: {y['description']}' for x, y in roles.items()}
            similarity_graph = SimilarityGraph(TfidfVectorizer, threshold=0.5)
            G = similarity_graph.detect_and_remove_similar_entries(graph_data)
            roles = {x: y for x, y in roles.items() if x in G.nodes()}

        return roles

    def generate_roles(self, n_roles, initial_role_names=None):
        all_roles = {}
        prev_roles = [] if initial_role_names is None else initial_role_names

        if n_roles <= 0:
            return all_roles
        while len(all_roles) < n_roles:
            prompt = read_prompts(self.prompts_conf.roles_agents, key='USER_GEN',
                                  context={'workspace': self.workspace, 'domain': self.domain,
                                           'prev_roles': prev_roles,
                                           'n_initial_roles': self.n_initial_roles}, logger=self.logger)
            try:
                response = run_agent_query(prompt=prompt, agent=self.roles_generation_agent, logger=self.logger,
                                           to_json=True,
                                           json_transform_keys=['roles', 'name'])
                validate_roles(response, required_fields=['name', 'description'], n_roles=0)
                all_roles.update(response)
                prev_roles = list(all_roles.keys())
            except Exception as e:
                self.logger.error(f"Error in generate_roles: {e}")
                continue

        return all_roles

    def judge_roles(self, input_roles):
        missing_roles = list(input_roles.keys())
        out = deepcopy(input_roles)
        passes_requirements = False

        while not passes_requirements:
            if missing_roles:
                self.logger.debug(f"Missing roles: {missing_roles}")
            prompt = read_prompts(self.prompts_conf.judge_agents, key='USER_ROLES_VERIF',
                                  context={'roles': str({name: input_roles[name] for name in missing_roles})},
                                  logger=self.logger)

            try:
                response = run_agent_query(prompt=prompt, agent=self.roles_verif_judge, logger=self.logger,
                                           to_json=True,
                                           json_transform_keys=['roles', 'name'])
            except json.decoder.JSONDecodeError as e:
                self.logger.error(f"JSONDecodeError in judge_roles run_agent_query: {type(e)}:{e}")
                continue
            except Exception as e:
                self.logger.error(f"Error in judge_roles run_agent_query: {type(e)}:{e}")
                continue

            new_missing_roles = []
            passes_requirements = True
            for role_name in missing_roles:
                try:
                    self.logger.debug(f"Checking feedback for role: {role_name}")
                    if role_name not in response:
                        new_missing_roles.append(role_name)
                        passes_requirements = False
                        continue
                    out[role_name]['feedback'] = response[role_name]['feedback']
                    out[role_name]['is_accepted'] = response[role_name]['is_accepted']
                except Exception as e:
                    new_missing_roles.append(role_name)
                    passes_requirements = False
                    self.logger.error(f"Error in judge_roles: {e}")

            missing_roles = new_missing_roles

        is_accepted = all(role['is_accepted'] for role in out.values())
        return is_accepted, out

    def generate_and_judge_initial_roles(self, logging=True):
        try:
            with open(self.object_storage_conf.roles, 'r') as f:
                curr_accepted_roles = json.load(f)
                curr_accepted_roles = self.normalize_roles(curr_accepted_roles, remove_similar=False)
        except FileNotFoundError as e:
            self.logger.error(f"Error in generate_and_judge_initial_roles: {e}")
            curr_accepted_roles = {}

        accepted_roles = curr_accepted_roles
        while True:
            initial_role_names = list(curr_accepted_roles.keys()) if curr_accepted_roles else None
            n_roles_to_generate = max(0, self.n_initial_roles - len(accepted_roles))
            if logging:
                self.logger.debug(f'Roles left to generate: {n_roles_to_generate}')
            if len(accepted_roles) >= self.n_initial_roles:
                break

            generated_roles = self.generate_roles(n_roles=n_roles_to_generate, initial_role_names=initial_role_names)
            if logging:
                self.logger.debug(f'Generated roles_dict: {generated_roles}\n\n')

            is_accepted, judged_roles = self.judge_roles(generated_roles)
            curr_accepted_roles = {name: role for name, role in judged_roles.items() if role['is_accepted']}
            if logging:
                self.logger.debug(f'Output for judge_roles: {judged_roles}, {is_accepted}')

            curr_accepted_roles = self.normalize_roles(curr_accepted_roles, remove_similar=True)

            accepted_roles.update(curr_accepted_roles)
            if logging:
                self.logger.debug(f'Accepted role names: {list(accepted_roles.keys())}\n\n')

        accepted_roles = self.normalize_roles(accepted_roles, remove_similar=False)
        save_to_disk(accepted_roles, self.object_storage_conf.roles)
        return accepted_roles
