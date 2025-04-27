import json

# File paths
states_file = 'output/finalized_jsons/scenarios_messages.json'
funcs_file = 'output/finalized_jsons/scenarios_messages_single1.json'
output_file = 'output/finalized_jsons/scenarios_messages_single1.json'

# Load the JSON files
with open(states_file, 'r') as f:
    scenarios_states = json.load(f)

with open(funcs_file, 'r') as f:
    scenarios_funcs = json.load(f)

# Add attack_vector to scenarios_funcs
for domain, workspaces in scenarios_states.items():
    for workspace, roles in workspaces.items():
        for role, role_data in roles.items():
            if 'scenarios' in role_data:
                for scenario_name, scenario_data in role_data['scenarios'].items():
                    attack_vector = scenario_data.get('policy')
                    if attack_vector:
                        # Navigate to the corresponding scenario in scenarios_funcs
                        try:
                            scenarios_funcs[domain][workspace][role]['scenarios'][scenario_name]['policy'] = attack_vector
                        except KeyError:
                            print(f"Scenario {scenario_name} not found in scenarios_funcs.json")

# Save the updated scenarios_funcs to a new file
with open(output_file, 'w') as f:
    json.dump(scenarios_funcs, f, indent=4)

print(f"Updated scenarios_funcs saved to {output_file}")