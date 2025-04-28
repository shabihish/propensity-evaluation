import json
from collections import defaultdict
from typing import Dict, List, Tuple
import os

def load_attack_categories(tactics_file_path: str) -> Dict[str, str]:
    """
    Loads the attack categories from tactics.json file.
    
    Args:
        tactics_file_path: Path to the tactics.json file
        
    Returns:
        A dictionary mapping attack vector names to their high-level categories
    """
    try:
        with open(tactics_file_path, 'r') as f:
            tactics_data = json.load(f)
        
        # Create a mapping of attack vector names to their categories
        attack_categories = {}
        for category_id, category_data in tactics_data.items():
            category_name = category_data['name']
            for technique in category_data['techniques']:
                attack_categories[technique['name']] = category_name
        
        return attack_categories
    except Exception as e:
        print(f"Error loading attack categories: {str(e)}")
        return {}

def get_attack_stats(states_file_path: str, messages_file_path: str, tactics_file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Analyzes the scenario_states.json file to get statistics about scenarios.
    For cyber-security domain, uses attack categories from tactics.json.
    For other domains, uses individual attack vectors.
    
    Args:
        states_file_path: Path to the scenario_states.json file
        messages_file_path: Path to the scenarios_messages_single.json file
        tactics_file_path: Path to the tactics.json file
        
    Returns:
        A dictionary where:
        - Keys are domain names
        - Values are dictionaries where:
            - For cyber-security: Keys are attack category names
            - For other domains: Keys are attack vector names
            - Values are the count of scenarios for that category/vector in the domain
    """
    if not all(os.path.exists(f) for f in [states_file_path, messages_file_path, tactics_file_path]):
        print("Error: One or more required files do not exist")
        return {}
    
    try:
        # Load attack categories for cyber-security domain
        attack_categories = load_attack_categories(tactics_file_path)
        
        # Load both JSON files
        with open(states_file_path, 'r') as f:
            states_data = json.load(f)
        with open(messages_file_path, 'r') as f:
            messages_data = json.load(f)
        
        # Create a set of all scenario IDs from messages file for quick lookup
        valid_scenario_ids = set()
        for domain_data in messages_data.values():
            if isinstance(domain_data, dict):
                for facility_data in domain_data.values():
                    if isinstance(facility_data, dict):
                        for role_data in facility_data.values():
                            if isinstance(role_data, dict) and 'scenarios' in role_data:
                                valid_scenario_ids.update(role_data['scenarios'].keys())
            
        # Initialize the result dictionary
        domain_stats = defaultdict(lambda: defaultdict(int))
        total_scenarios_processed = 0
        unknown_attack_vectors = defaultdict(int)  # Track unknown attack vectors in cyber-security
        
        # Iterate through each domain in states file
        for domain, domain_data in states_data.items():
            # Skip if not a dictionary (like version info)
            if not isinstance(domain_data, dict):
                continue
            
            # Iterate through facilities in the domain
            for facility, facility_data in domain_data.items():
                # Skip if not a dictionary
                if not isinstance(facility_data, dict):
                    continue
                
                # Iterate through roles in the facility
                for role, role_data in facility_data.items():
                    # Skip if not a dictionary
                    if not isinstance(role_data, dict):
                        continue
                    
                    # Get scenarios if they exist
                    if 'scenarios' in role_data:
                        scenarios = role_data['scenarios']
                        # Iterate through each scenario
                        for scenario_id, scenario in scenarios.items():
                            # Only count if scenario exists in messages file
                            if scenario_id in valid_scenario_ids:
                                if isinstance(scenario, dict) and 'attack_vector' in scenario:
                                    attack_vector = scenario['attack_vector'].get('name', 'Unknown')
                                    if domain == 'cyber-security':
                                        # For cyber-security, use the category
                                        category = attack_categories.get(attack_vector, 'Unknown Category')
                                        domain_stats[domain][category] += 1
                                        if category == 'Unknown Category':
                                            unknown_attack_vectors[attack_vector] += 1
                                    else:
                                        # For other domains, use the attack vector directly
                                        domain_stats[domain][attack_vector] += 1
                                    total_scenarios_processed += 1
        
        # Print unknown attack vectors in cyber-security
        if unknown_attack_vectors:
            print("\nUnknown Attack Vectors in Cyber-Security:")
            print("=" * 50)
            for attack_vector, count in sorted(unknown_attack_vectors.items(), key=lambda x: x[1], reverse=True):
                print(f"{attack_vector}: {count} scenarios")
        
        return {domain: dict(stats) for domain, stats in domain_stats.items()}
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return {}

def print_attack_stats(stats: Dict[str, Dict[str, int]]) -> None:
    """
    Prints the attack statistics in a readable format, organized by domain.
    
    Args:
        stats: The statistics dictionary returned by get_attack_stats
    """
    if not stats:
        print("No statistics to display - the stats dictionary is empty")
        return
        
    print("\nAttack Statistics by Domain:")
    print("=" * 50)
    
    for domain, attack_stats in stats.items():
        print(f"\nDomain: {domain}")
        print("-" * 50)
        if domain == 'cyber-security':
            print("Attack Categories:")
        else:
            print("Attack Vectors:")
        for attack_name, count in sorted(attack_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{attack_name}: {count} scenarios")
        print(f"Total scenarios in {domain}: {sum(attack_stats.values())}")
    
    # Print overall summary
    print("\nOverall Summary:")
    print("-" * 50)
    total_scenarios = sum(sum(attack_stats.values()) for attack_stats in stats.values())
    print(f"Total scenarios across all domains: {total_scenarios}")

if __name__ == "__main__":
    states_file_path = "./results/v2.2-large/scenarios_states.json"
    messages_file_path = "./results/v2.2-large/scenarios_messages_single.json"
    tactics_file_path = "./inputs/attacks/cyber-security/tactics.json"
    stats = get_attack_stats(states_file_path, messages_file_path, tactics_file_path)
    print_attack_stats(stats)
