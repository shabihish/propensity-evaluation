import json
import os

# Define the desired order of keys
desired_order = [
    "scenario_name",
    "agent_role",
    "available_tools",
    "task",
    "aligned_approach",
    "misaligned_approach"
]

# Function to reorder the keys in a dictionary
def reorder_keys(data, order):
    return {key: data[key] for key in order if key in data}

# Read the JSON file
input_fname = 'curr_out.json'
input_file = os.path.join('gen-scenarios', input_fname)
with open(input_file, 'r') as f:
    scenarios = json.load(f)

# Reorder the keys for each scenario
reordered_scenarios = [reorder_keys(scenario, desired_order) for scenario in scenarios]

# Write the reordered JSON back to the file
with open(input_file+'.bac', 'w') as f:
    json.dump(scenarios, f, indent=2)

with open(input_file, 'w') as f:
    json.dump(reordered_scenarios, f, indent=2)