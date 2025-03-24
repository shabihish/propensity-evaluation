import json
import random
from copy import deepcopy

from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer

from agents.agent import Agent
from .graph_utils import SimilarityGraph
from .utils import json_obj_list_to_dict
from .utils import read_prompts, run_agent_query, check_for_missing_fields, \
    load_output_schemas


def get_valid_scenarios(roles: dict, required_fields: list, min_scenarios_per_role: int = None):
    failed_roles = check_for_missing_fields(roles, required_fields)
    if failed_roles:
        roles = {k: v for k, v in roles.items() if v['name'] not in failed_roles}
    if min_scenarios_per_role:
        roles = {k: v for k, v in roles.items() if len(v['scenarios']) >= min_scenarios_per_role}
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


class AttackVectorsPoolManager:
    def __init__(self, attack_vectors_per_role: dict, n_samples: int = 0):
        self.n_samples = n_samples
        self.original_vectors = {role: list(vectors) for role, vectors in attack_vectors_per_role.items()}
        self.remaining_vectors = deepcopy(self.original_vectors)
        self.current_vectors = {}

        self.sample_counts = {role: 0 for role in attack_vectors_per_role}

        if n_samples > 0:
            self.max_original_vector_len = n_samples
        else:
            self.max_original_vector_len = max(len(vectors) for vectors in self.original_vectors.values())

    def pop_next_vectors(self) -> dict:
        self.flush_current_vectors()
        for role in self.remaining_vectors:
            _ = self.pop_next_vector(role)
            assert role in self.current_vectors
        return self.current_vectors

    def pop_next_vector(self, role: str):
        if role in self.remaining_vectors:
            if not self.remaining_vectors[role]:
                return None
            if 0 < self.n_samples <= self.sample_counts[role]:
                return None
            if self.n_samples == 0:
                vector = self.remaining_vectors[role].pop(0)
            else:
                vector = random.choice(self.remaining_vectors[role])
                self.remaining_vectors[role].remove(vector)
                self.sample_counts[role] += 1
            self.current_vectors[role] = vector
            return vector
        return None

    def flush_current_vectors(self):
        self.current_vectors.clear()

    def revert_failed_vector(self, role: str):
        if role in self.current_vectors:
            self.remaining_vectors[role].append(self.current_vectors.pop(role))
            if self.n_samples > 0:
                self.sample_counts[role] -= 1

    def has_remaining_vectors(self) -> bool:
        return any(len(vectors) > 0 for vectors in self.remaining_vectors.values())


class ScenarioManager:
    def __init__(self, api_conf, logger, workspace_name: str, workspace_desc: str, workspace_alternative_forms: list,
                 domain_name, domain_desc: str,
                 domain_alternative_forms: list, prompts_conf: DictConfig,
                 output_schemas_conf: DictConfig, object_storage_conf: DictConfig, temperature,
                 min_initial_scenarios_per_role: int, min_chosen_scenarios_per_role: int, generation_batch_size: int):
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

        self.min_initial_scenarios_per_role = min_initial_scenarios_per_role
        self.min_chosen_scenarios_per_role = min_chosen_scenarios_per_role
        self.batch_size = generation_batch_size

        self.scenarios_generation_agent = self._init_scenarios_generation_agent()
        self.scenarios_verif_judge = self._init_scenarios_verif_judge()

        self.similarity_graph = SimilarityGraph(TfidfVectorizer, threshold=0.5)

    def _init_scenarios_generation_agent(self):
        general_body = read_prompts(self.prompts_conf.scenarios_general_body, key='SYS_GEN',
                                    context={'workspace': self.workspace, 'workspace_desc': self.workspace_desc,
                                             'domain': self.domain, 'domain_desc': self.domain_desc},
                                    logger=self.logger)

        sys_prompt = read_prompts(self.prompts_conf.scenarios_agents_states, key='SYS_GEN',
                                  context={'general_body': general_body,
                                           'n_scenarios': self.min_initial_scenarios_per_role}, logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_states)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def _init_scenarios_verif_judge(self):
        sys_prompt = read_prompts(self.prompts_conf.judge_agents, key='SYS_SCENARIOS_STATES_VERIF',
                                  context={'workspace': self.workspace, 'workspace_desc': self.workspace_desc,
                                           'domain': self.domain, 'domain_desc': self.domain_desc},
                                  logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.judge_scenarios)
        return Agent(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def remove_similar_scenarios(self, roles_with_scenarios: dict, min_chosen_scenarios_per_role: int):
        out_roles = deepcopy(roles_with_scenarios)
        for role_k, role_v in roles_with_scenarios.items():
            curr_scenarios = role_v['scenarios']
            # Form the similarity graph and remove similar entries
            graph_data = {scenario['name']: scenario['description'] for scenario in curr_scenarios}
            similarity_graph = SimilarityGraph(TfidfVectorizer, threshold=0.5)
            G = similarity_graph.detect_and_remove_similar_entries(graph_data)
            out_roles[role_k]['scenarios'] = [x for x in curr_scenarios if x['name'] in G.nodes()]

        out_roles = {k: v for k, v in out_roles.items() if len(v['scenarios']) >= min_chosen_scenarios_per_role}
        return out_roles

    def generate_scenarios(self, input_roles):
        roles_with_scenarios = deepcopy(input_roles)
        roles_to_process = list(input_roles.keys())
        batch_size = self.batch_size
        i = 0

        while roles_to_process:
            i += 1
            # Try at least 3 times for each of the batches
            if i > 3 * max(1, len(roles_to_process) // batch_size):
                self._log_error(
                    f"Exiting generator after {i - 1} batch iterations. Still missing roles: {roles_to_process}")
                for role in roles_to_process:
                    roles_with_scenarios.pop(role)
                break

            batch_roles = self._get_batch_roles(roles_to_process, batch_size)
            prompt = self._create_generation_prompt(batch_roles, input_roles)

            try:
                response = self._run_generation_agent(prompt)
                response = self._process_generation_response(response, input_roles)
                self._update_roles_with_scenarios(response, roles_with_scenarios, roles_to_process)
            except json.JSONDecodeError as e:
                prev_batch_size = batch_size
                batch_size = max(1, batch_size // 2)
                self.logger.warning(
                    f"Error in generate_scenarios; Reducing batch size from {prev_batch_size} to {batch_size}.")
            except Exception as e:
                self._log_error(f"Error in generate_scenarios: {e}")

        return roles_with_scenarios

    def _get_batch_roles(self, roles_to_process, batch_size):
        batch_roles = roles_to_process[:batch_size]
        random.shuffle(batch_roles)
        return batch_roles

    def _create_generation_prompt(self, batch_roles, input_roles):
        return read_prompts(self.prompts_conf.scenarios_agents_states, key='USER_GEN',
                            context={'roles': {role: input_roles[role] for role in batch_roles}},
                            logger=self.logger)

    def _run_generation_agent(self, prompt):
        return run_agent_query(prompt=prompt, agent=self.scenarios_generation_agent, logger=self.logger, to_json=True,
                               json_transform_keys=['roles', 'name'])

    def _process_generation_response(self, response, input_roles):
        response = get_valid_scenarios(response, required_fields=['name', 'scenarios'],
                                       min_scenarios_per_role=self.min_initial_scenarios_per_role)
        response = self.remove_similar_scenarios(response,
                                                 min_chosen_scenarios_per_role=self.min_chosen_scenarios_per_role)
        for role in response:
            response[role]['scenarios'] = json_obj_list_to_dict(response[role]['scenarios'], 'name')
        return response

    def _update_roles_with_scenarios(self, response, roles_with_scenarios, roles_to_process):
        for role in response.values():
            if role['name'] in roles_with_scenarios:
                roles_with_scenarios[role['name']].update(role)
                roles_to_process.remove(role['name'])

    def _log_error(self, message):
        if self.logger:
            self.logger.error(message)

    def judge_scenarios(self, input_scenarios: dict):
        roles_to_process = list(input_scenarios.keys())
        out = deepcopy(input_scenarios)
        batch_size = self.batch_size
        i = 0

        while roles_to_process:
            i += 1
            # Try at least 3 times for each of the batches
            if i > 3 * max(1, len(roles_to_process) // batch_size):
                self._log_error(
                    f"Exiting generator after {i - 1} batch iterations. Still missing roles: {roles_to_process}")
                for role in roles_to_process:
                    out.pop(role)
                break

            batch_roles = self._get_batch_roles(roles_to_process, batch_size)
            prompt = self._create_judgment_prompt(batch_roles, input_scenarios)

            try:
                response = self._run_judgment_agent(prompt)
                response = self._process_judgment_response(response, input_scenarios)
                self._update_judged_scenarios(response, out, roles_to_process)
            except json.JSONDecodeError as e:
                prev_batch_size = batch_size
                batch_size = max(1, batch_size // 2)
                self.logger.warning(
                    f"Error in generate_scenarios; Reducing batch size from {prev_batch_size} to {batch_size}.")
            except Exception as e:
                self._log_error(f"Error in judge_scenarios: {e}")

        return out

    def _create_judgment_prompt(self, batch_roles, input_scenarios):
        return read_prompts(self.prompts_conf.judge_agents, key='USER_SCENARIOS_VERIF',
                            context={'roles': str({name: input_scenarios[name] for name in batch_roles})},
                            logger=self.logger)

    def _run_judgment_agent(self, prompt):
        return run_agent_query(prompt=prompt, agent=self.scenarios_verif_judge, logger=self.logger, to_json=True,
                               json_transform_keys=['roles', 'name'])

    def _process_judgment_response(self, response, input_scenarios):
        response = get_valid_scenarios(response, required_fields=['name', 'scenarios'])
        response = {k: v for k, v in response.items() if
                    set(x['name'] for x in v['scenarios']) == set(input_scenarios[k]['scenarios'].keys())}
        for role in response:
            response[role]['scenarios'] = json_obj_list_to_dict(response[role]['scenarios'], 'name')
        return response

    def _update_judged_scenarios(self, response, out, roles_to_process):
        for role in response.values():
            if role['name'] in out:
                out[role['name']].update(role)
                roles_to_process.remove(role['name'])

    def generate_and_judge_scenarios(self, input_roles: dict, grounding_attack_vectors: dict, grounding_n_samples: int,
                                     logging=True):
        accepted_scenarios = {}
        missing_scenarios = list(set(input_roles.keys()))
        assert all(x in grounding_attack_vectors and isinstance(grounding_attack_vectors[x], list) and len(
            grounding_attack_vectors[x]) >= 1 for x in missing_scenarios)
        attacks_pool_manager = AttackVectorsPoolManager(grounding_attack_vectors, grounding_n_samples)

        n_tries_for_role = 0
        while missing_scenarios:
            if n_tries_for_role >= attacks_pool_manager.max_original_vector_len + 10:
                self.logger.error(
                    f"Exceeded max number of tries for the root-lvl for workspace {self.workspace}. Exiting. "
                    f"Unprocessed roles: {missing_scenarios}")
                break
            n_tries_for_role += 1

            roles_to_process = self._prepare_roles_for_processing(input_roles, missing_scenarios, attacks_pool_manager,
                                                                  logging)
            if logging:
                self.logger.debug(f"Current root-lvl roles to process keys: {roles_to_process.keys()}")

            if not roles_to_process:
                continue

            generated_scenarios = self.generate_scenarios(roles_to_process)
            if logging:
                self.logger.debug(f"Generated scenarios: {json.dumps(generated_scenarios, indent=2)}")

            judged_scenarios = self.judge_scenarios(generated_scenarios)
            if logging:
                self.logger.debug(f"Judged scenarios: {json.dumps(judged_scenarios, indent=2)}")

            self._update_accepted_scenarios(judged_scenarios, input_roles, generated_scenarios, accepted_scenarios,
                                            attacks_pool_manager, logging)
            if logging:
                self.logger.debug(
                    f"Currently accepted roles with number of scenarios: { {r: len(accepted_scenarios[r]['scenarios']) for r in accepted_scenarios} }")
        return accepted_scenarios

    def _prepare_roles_for_processing(self, input_roles, missing_scenarios, attacks_pool_manager, logging):
        roles_to_process = {}
        attacks_pool_manager.flush_current_vectors()

        for role in missing_scenarios:
            next_attack_vector = attacks_pool_manager.pop_next_vector(role)
            if next_attack_vector:
                roles_to_process[role] = deepcopy(input_roles[role])
                roles_to_process[role]['attack_vector'] = next_attack_vector
            else:
                if logging:
                    self.logger.debug(f"Role {role} has no more attack vectors left. Removing from missing roles.")
                missing_scenarios.remove(role)
        return roles_to_process

    def _update_accepted_scenarios(self, judged_scenarios, input_roles, generated_scenarios, accepted_scenarios,
                                   attacks_pool_manager, logging):
        for role_k, role_v in judged_scenarios.items():
            accepted_scenarios_for_role = [curr_scenario for curr_scenario in list(role_v['scenarios'].values()) if
                                           curr_scenario and curr_scenario['acceptable']]

            if accepted_scenarios_for_role:
                # missing_scenarios.remove(role_k)
                if logging:
                    self.logger.debug(f'Accepted final scenario for role {role_k}: {accepted_scenarios_for_role}')
                self._store_accepted_scenarios(role_k, accepted_scenarios_for_role, input_roles, generated_scenarios,
                                               judged_scenarios, accepted_scenarios)
            else:
                attacks_pool_manager.revert_failed_vector(role_k)
                if logging:
                    self.logger.debug(f"All final scenarios rejected for role {role_k}. Reverted attack vector.")

    def _store_accepted_scenarios(self, role_k, accepted_scenarios_for_role, input_roles, generated_scenarios,
                                  judged_scenarios, accepted_scenarios):
        for curr_scenario in accepted_scenarios_for_role:
            scen_name = curr_scenario['name']
            initial_curr = input_roles[role_k]
            generated_curr = generated_scenarios[role_k]
            judged_curr = judged_scenarios[role_k]
            if role_k not in accepted_scenarios:
                accepted_scenarios[role_k] = deepcopy(initial_curr)
                accepted_scenarios[role_k]['scenarios'] = {}
            accepted_scenarios[role_k]['scenarios'].update(
                {curr_scenario['name']: generated_curr['scenarios'][scen_name]})
            accepted_scenarios[role_k]['scenarios'][scen_name]['acceptable'] = judged_curr['scenarios'][scen_name][
                'acceptable']
            accepted_scenarios[role_k]['scenarios'][scen_name]['feedback'] = judged_curr['scenarios'][scen_name][
                'feedback']
            accepted_scenarios[role_k]['scenarios'][scen_name]['attack_vector'] = judged_curr['attack_vector']
            accepted_scenarios[role_k].pop('attack_vector', None)
