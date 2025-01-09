import re
import json
from copy import deepcopy
from utils.regex import remove_duplicate_names_by_prefix, apply_to_all_keys
from sim.modules.utils import read_prompts


def generate_roles(simulator, n_roles, initial_role_names=None):
    all_roles = {}
    prev_roles = [] if initial_role_names is None else initial_role_names

    if n_roles <= 0:
        return all_roles
    while len(all_roles) < n_roles:
        prompt = read_prompts(simulator.cfg.prompts.roles_agents, key='USER_GEN',
                              context={**vars(simulator), 'prev_roles': prev_roles}, logger=simulator.logger)
        try:
            response = simulator.run_agent_query(prompt, simulator.roles_generation_agent, to_json=True,
                                                 json_transform_keys=['roles', 'name'])
            simulator.validate_roles(response, required_fields=['name', 'description'], n_roles=0)
            all_roles.update(response)
            prev_roles = list(all_roles.keys())
        except Exception as e:
            simulator.logger.error(f"Error in generate_roles: {e}")
            continue

    return all_roles


def judge_roles(simulator, input_roles):
    missing_roles = list(input_roles.keys())
    out = deepcopy(input_roles)
    passes_requirements = False

    while not passes_requirements:
        if missing_roles:
            simulator.logger.debug(f"Missing roles: {missing_roles}")
        prompt = read_prompts(simulator.cfg.prompts.judge_agents, key='USER_ROLES_VERIF',
                              context={'roles': str({name: input_roles[name] for name in missing_roles})},
                              logger=simulator.logger)
        passes_requirements = True
        response = simulator.run_agent_query(prompt, simulator.roles_verif_judge, to_json=True,
                                             json_transform_keys=['roles', 'name'])

        new_missing_roles = []
        for role_name in missing_roles:
            try:
                simulator.logger.debug(f"Checking feedback for role: {role_name}")
                if role_name not in response:
                    new_missing_roles.append(role_name)
                    passes_requirements = False
                    continue
                out[role_name]['feedback'] = response[role_name]['feedback']
                out[role_name]['is_accepted'] = response[role_name]['is_accepted']
            except Exception as e:
                new_missing_roles.append(role_name)
                passes_requirements = False
                simulator.logger.error(f"Error in judge_roles: {e}")

        missing_roles = new_missing_roles

    is_accepted = all(role['is_accepted'] for role in out.values())
    return is_accepted, out
