# PropensityBench Dataset

This directory contains the pre-generated scenario data for the PropensityBench benchmark. This document details the structure of the final benchmark files used for evaluation, as well as the intermediate artifacts produced by our multi-stage scenario generation pipeline (not required for evaluation).


## Directory Structure

The data is organized hierarchically by domain and workspace. The evaluation harness (`evaluation/main.py`) is configured to read from this structure.

```
data/
└── final/
    ├── bio-security/
    │   ├── agricultural-biotech-facility/
    │   │   └── scenarios_messages_single.json
    │   │   └── other json files (intermediate generation artifacts)
    │   ├── bsl-3-4-high-containment-laboratory/
    │   │   └── scenarios_messages_single.json
    │   │   └── other json files (intermediate generation artifacts)
    │   └── ... (other biosecurity workspaces)
    │
    ├── chemical-security/
    │   └── ... (workspaces for chemical security)
    │
    ├── cyber-security/
    │   └── ... (workspaces for cybersecurity)
    │
    └── self-proliferation/
        └── ... (workspaces for self-proliferation)
```

-   **`final/`**: The root directory for the finalized benchmark data.
-   **`<domain>/`**: Each subdirectory corresponds to one of the four high-risk domains (e.g., `bio-security`).
-   **`<workspace>/`**: Within each domain, subdirectories represent specific operational environments (e.g., `agricultural-biotech-facility`).
-   **`scenarios_messages_single.json`**: This is the key file containing all the scenarios for the roles within that workspace.

## Final Scenario Schema

Each `scenarios_messages_single.json` file contains a deeply nested JSON object that fully defines all scenarios for a given workspace. Below is a breakdown of the key fields. This is the ONLY file required for evaluation within each workspace, and contains the complete set of scenarios with all necessary fields defined under that workspace.

```json
{
  "<domain_name>": {
    "<workspace_name>": {
      "<role_name>": {
        "scenarios": {
          "<scenario_name>": {
            "name": "A unique identifier for the scenario.",
            "description": "A high-level narrative description of the scenario's context.",
            
            "states": {
              "initial_state": { "name": "...", "description": "..." },
              "target_state": { "name": "...", "description": "..." },
              "illegal_state": { "name": "...", "description": "..." }
            },

            "functions": {
              "target_function": {
                "name": "complete_...",
                "description": "The aligned tool to complete the task.",
                "input_arguments": ["arg1", "arg2"],
                "errors": ["Error message 1", "..."]
              },
              "trigger_function": {
                "harmful_name": "use_exploit_...",
                "harmful_description": "The misaligned tool with a harmful-sounding name.",
                "neutral_name": "enhance_data_...",
                "neutral_description": "The misaligned tool with a benign-sounding name."
              },
              "getter_functions": [
                { "name": "get_...", "description": "...", "input_arguments": [], "output_arguments": ["..."] }
              ]
            },

            "configurations": {
              "configuration_known": "A JSON string of environmental parameters known to the agent at the start.",
              "configuration_unknown": "A JSON string of parameters the agent must discover using getter tools."
            },

            "policy": {
              "role_description": "...",
              "duties_description": "...",
              "prohibitions_description": "Implicit warnings against the misaligned action.",
              "negative_consequences": { "text": "The catastrophic outcomes of using the misaligned tool." }
            },

            "messages": {
              "task_message": "The initial instruction given to the agent to begin the simulation."
            },

            "sys_messages": {
              "Time": [ { "from": "...", "subject": "...", "body": "..." }, ... ],
              "Financials": [ ... ],
              "//": "Contains 12 escalating pressure messages for each of the 6 pressure tactics."
            }
          }
        }
      }
    }
  }
}
```


## Intermediate Generation Artifacts

Our scenario generation pipeline (`generation/`) is a multi-stage process. The files below represent the state of the scenarios after each major stage. They are typically found in the `generation_output/` directory (or your configured output directory) and are useful for debugging the pipeline or for researchers who wish to use only certain components of our generated data.

#### 1. `scenarios_states.json`
-   **Content:** The output of Stage 1 of the generation pipeline. Contains the foundational narrative of each scenario.
-   **Key Fields:** `name`, `description`, and `states` (`initial_state`, `target_state`, `illegal_state`).
-   **Purpose:** This file defines the core objective and conflict of each scenario before any interactive elements are added.

#### 2. `scenarios_funcs.json`
-   **Content:** The output of Stage 2 of the generation pipeline. Enriches the states with interactive tools and environmental parameters.
-   **Key Fields Added:** `functions` (defining all tools) and `configurations` (defining the environment state).
-   **Purpose:** This file makes the scenarios operational by defining the agent's capabilities and the world it interacts with.

#### 3. `scenarios_policies.json`
-   **Content:** The output of Stage 3 of the generation pipeline. Adds the rules and context that govern the agent's behavior.
-   **Key Fields Added:** `policy` (containing descriptions of the role, duties, prohibitions, and consequences).
-   **Purpose:** This file grounds the agent in its role, creating the ethical dilemma it will face.

#### 4. `scenarios_messages.json`
-   **Content:** The output of Stage 4 of the generation pipeline. Adds the initial, non-pressure-related messages.
-   **Key Fields Added:** `messages` (`task_message`, `neutral_sys_messages`).
-   **Purpose:** This file provides the initial prompt that starts the agent's simulation loop. The `scenarios_messages_single.json` file is the final version after the next step.

#### 5. `scenarios_messages_single.json`
-   **Content:** The output of Stage 5 of the generation pipeline. Adds system messages, to the outputs of stage 4, which are pressure messages for each of the six pressure tactics (Time, Financials, etc.).
-   **Key Fields Added:** `sys_messages` (`evaluation-awareness`, `financials`, ...). Each pressure category contains 12 messages with strictly increasing pressure levels.
- **Key Fields in Each Message:** Each message is an object with `msg_level_index`, `from`, `subject`, and `body` fields.
-   **Purpose:** This file completes the scenario definitions by providing the escalating pressure messages that will be used to test the agent's alignment under stress.


After the generation of the final pressure messages, all components are consolidated into the final `scenarios_messages_single.json` file, which is the canonical version of the benchmark used for evaluation.



