from dotenv import load_dotenv

load_dotenv()

import hydra
import datasets
import logging
from colorama import Fore, Style
import hydra
from utils.colors import BaseColoredFormatter
import logging

from sim.pipeline_sim import PipelineSimulator

WORKSPACE_NAME = 'datacenter'
WORKSPACE_ALTERNATIVE_FORMS = ['data-center', 'data center']
# WORKSPACE_ALTERNATIVE_FORMS = ['biology-lab', 'bio lab', 'biolab', 'bio-lab']
# WORKSPACE_NAME = 'Genome Sequencing Facility'
# WORKSPACE_ALTERNATIVE_FORMS = ['genome-sequencing-facility', 'genome sequencing facility']
# WORKSPACE_DESC = ('A laboratory equipped with high-throughput sequencing technologies (e.g., Illumina, PacBio) and '
#                   'bioinformatics infrastructure for analyzing large-scale genomic data from various organisms.  This '
#                   'includes data processing, storage, and analysis pipelines for tasks like genome assembly, '
#                   'variant calling, and gene expression analysis.')
WORKSPACE_DESC = 'A datacenter is a facility composed of networked computers and storage used to organize, process'

# DOMAIN_NAME = 'biology'
# DOMAIN_ALTERNATIVE_FORMS = ['bio', 'biological systems', 'biological-systems']

DOMAIN_NAME = 'cybersecurity'
DOMAIN_ALTERNATIVE_FORMS = ['cyber-security', 'cyber security']

# def setup_logger(name='colored_logger', log_file='app.log', level=logging.DEBUG):
#     """Set up a logger with both file and console handlers"""
#
#     # Create logger
#     logger = logging.getLogger(name)
#     # logger.setLevel(level)
#     #
#     # # Create handlers
#     console_handler = logging.StreamHandler()
#     # file_handler = logging.FileHandler(log_file)
#     #
#     # # Set levels
#     console_handler.setLevel(level)
#     # file_handler.setLevel(level)
#     #
#     # # Create formatters
#     console_formatter = BaseColoredFormatter(
#         '%(asctime)s - %(levelname)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
#     # file_formatter = logging.Formatter(
#     #     '%(asctime)s - %(levelname)s - %(message)s',
#     #     datefmt='%Y-%m-%d %H:%M:%S'
#     # )
#     #
#     # # Add formatters to handlers
#     console_handler.setFormatter(console_formatter)
#     # file_handler.setFormatter(file_formatter)
#     #
#     # # Add handlers to logger
#     logger.addHandler(console_handler)
#     # logger.addHandler(file_handler)
#
#     return logger


import logging
from colorama import Fore, Style


# class BaseColoredFormatter(logging.Formatter):
#     def format(self, record):
#         levelno = record.levelno
#         if levelno == 'DEBUG':
#             record.levelname = f"{Fore.BLUE}{record.levelname}{Style.RESET_ALL}"
#         elif levelno == 'INFO':
#             record.levelname = f"{Fore.GREEN}{record.levelname}{Style.RESET_ALL}"
#         elif levelno == 'WARNING':
#             record.levelname = f"{Fore.YELLOW}{record.levelname}{Style.RESET_ALL}"
#         elif levelno == 'ERROR':
#             record.levelname = f"{Fore.RED}{record.levelname}{Style.RESET_ALL}"
#         elif levelno == 'CRITICAL':
#             record.levelname = f"{Fore.MAGENTA}{record.levelname}{Style.RESET_ALL}"
#         return super().format(record)


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
    sim = PipelineSimulator(cfg, logger, workspace_name=WORKSPACE_NAME, workspace_desc=WORKSPACE_DESC,
                            workspace_alternative_forms=WORKSPACE_ALTERNATIVE_FORMS,
                            domain_name=DOMAIN_NAME, domain_alternative_forms=DOMAIN_ALTERNATIVE_FORMS)

    sim.run()


# Example usage
if __name__ == "__main__":
    # Load environment variables
    main()
