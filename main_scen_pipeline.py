from dotenv import load_dotenv

load_dotenv()
import hydra
import os
import json
from utils.colors import BaseColoredFormatter
import logging
from copy import deepcopy
from pipeline.scenarios_pipeline import PipelineScenarios


def setup_logger(log_file='app.log'):
    logger = logging.getLogger()
    logging.getLogger().setLevel(logging.DEBUG)
    for logger_name in logging.root.manager.loggerDict:
        print(f'Muting logs from {logger_name}')
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    print()

    ch = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    ch.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    formatter = BaseColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    for handler in logging.root.handlers[:]:
        print(f'Removing logging handler: {handler}')
        logging.root.removeHandler(handler)

    logger.addHandler(ch)
    logger.addHandler(file_handler)
    return logger


def init_attack_vectors():
    attack_vectors = {}
    base_dir = 'inputs/attacks'

    # Scan all subdirectories in inputs/attacks
    for domain in os.listdir(base_dir):
        domain_path = os.path.join(base_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        # Look for attacks.json file in each subdirectory
        attacks_file = os.path.join(domain_path, 'attacks.json')
        if os.path.exists(attacks_file):
            with open(attacks_file, 'r') as f:
                attack_vectors[domain] = json.load(f)

    return attack_vectors


def get_attack_vectors_for_roles(roles, domain_attack_vectors: dict, logger):
    role_attack_vectors = {}
    for role_k, role_v in roles.items():
        if not role_v['attack_vectors']:
            role_attack_vectors[role_k] = list(domain_attack_vectors.values())
        else:
            role_attack_vectors[role_k] = [domain_attack_vectors[av] for av in role_v['attack_vectors']]

    return role_attack_vectors


def remove_keys_from_dictv(roles, keys: list):
    out = deepcopy(roles)
    for k, v in roles.items():
        for key in keys:
            if key in v:
                out[k].pop(key)

    return out


@hydra.main(config_path="configs", config_name="pipeline", version_base="1.2")
def main(cfg) -> None:
    logger = setup_logger()

    loaded_workspaces = [
        'bio-security.Agricultural Biotech Facility',
        'bio-security.Pandemic Response & Biosecurity Center',
        'bio-security.BSL-3/BSL-4 High-Containment Laboratory',
        'bio-security.Open-Access DIY Biohacking Lab',

        'chemical-security.Agricultural Chemical Development Facility',
        'chemical-security.Independent Contract Research Organization',
        'chemical-security.Materials Science Innovation Center',
        'chemical-security.Pharmaceutical Development Company',

        'cyber-security.Datacenter',
        'cyber-security.Enterprise Cybersecurity Solutions and Threat Mitigation Provider',
        'cyber-security.Confidential Legal Operations and Data Management Firm',
        'cyber-security.Advanced Space Exploration and Telemetry Command Center',
    ]
    # Load attack vectors for all domains
    attack_vectors = init_attack_vectors()

    base_inputs_dir = 'inputs/workspaces'
    with open(os.path.join(base_inputs_dir, 'domains.json'), 'r') as file:
        domain_data = json.load(file)

    for domain_name in os.listdir(base_inputs_dir):
        # Check for the chosen file to be of dir type
        if not os.path.isdir(os.path.join(base_inputs_dir, domain_name)):
            continue
        assert domain_name in attack_vectors, f'No attack vectors found for domain {domain_name}. Available domains: {attack_vectors.keys()}'
        domain_desc = domain_data[domain_name].get('description')
        domain_alternative_forms = domain_data[domain_name].get('alternative_forms')
        for filename in os.listdir(os.path.join(base_inputs_dir, domain_name)):
            if filename.endswith('.json'):
                filepath = os.path.join(base_inputs_dir, domain_name, filename)
                with open(filepath, 'r') as file:
                    workspace_data = json.load(file)
                    workspace_name = workspace_data.get('name')
                    workspace_desc = workspace_data.get('description')
                    workspace_alternative_forms = workspace_data.get('alternative_forms')
                    loaded_roles = workspace_data.get('roles', {})

                if loaded_workspaces and f'{domain_name}.{workspace_name}' not in loaded_workspaces:
                    logger.info(f'Skipping workspace {domain_name}.{workspace_name}')
                    continue

                prepared_roles = remove_keys_from_dictv(loaded_roles, ['attack_vectors'])
                current_avs = get_attack_vectors_for_roles(loaded_roles, attack_vectors[domain_name], logger)

                sim = PipelineScenarios(cfg, logger, workspace_name=workspace_name, workspace_desc=workspace_desc,
                                        workspace_alternative_forms=workspace_alternative_forms,
                                        domain_name=domain_name, domain_desc=domain_desc,
                                        domain_alternative_forms=domain_alternative_forms)

                sim.run(roles=prepared_roles,
                        grounding_attack_vectors=current_avs,
                        grounding_n_samples=3 if 'cyber' in domain_name else 0)


# Example usage
if __name__ == "__main__":
    # Load environment variables
    main()
