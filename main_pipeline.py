from dotenv import load_dotenv

load_dotenv()

import json
import os
import hydra
from utils.colors import BaseColoredFormatter
import logging


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


@hydra.main(config_path="configs", config_name="pipeline", version_base="1.2")
def main(cfg) -> None:
    logger = setup_logger()

    # sim = Pipeline(cfg, logger, workspace_name=WORKSPACE_NAME, workspace_desc=WORKSPACE_DESC,
    #                         workspace_alternative_forms=WORKSPACE_ALTERNATIVE_FORMS,
    #                         domain_name=DOMAIN_NAME, domain_alternative_forms=DOMAIN_ALTERNATIVE_FORMS)
    #
    # sim.run()

    base_inputs_dir = 'inputs/workspaces'
    with open(os.path.join(base_inputs_dir, 'domains.json'), 'r') as file:
        domain_data = json.load(file)

    for domain_name in os.listdir(base_inputs_dir):
        if not os.path.isdir(os.path.join(base_inputs_dir, domain_name)):
            continue
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
                    roles = workspace_data.get('roles', {})

                    # print(f'{workspace_name}:\troles: {len(roles.keys())}')
                    # print(workspace_name, workspace_desc, workspace_alternative_forms, '\n')
                    # sim = Pipeline(cfg, logger, workspace_name=workspace_name, workspace_desc=workspace_desc,
                    #                workspace_alternative_forms=workspace_alternative_forms,
                    #                domain_name=domain_name, domain_desc=domain_desc,
                    #                domain_alternative_forms=domain_alternative_forms)
                    # sim.run()


# Example usage
if __name__ == "__main__":
    # Load environment variables
    main()
