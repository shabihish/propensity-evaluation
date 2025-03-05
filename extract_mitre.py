import json
import re

from mitreattack.stix20 import MitreAttackData

valid_techniques = {
    'Initial Access': ['T1659', 'T1133', 'T1566', 'T1199', 'T1078'],
    'Execution': ['T1059', 'T1203', 'T1559', 'T1106', 'T1569', 'T1129'],
    'Persistence': ['T1098', 'T1136', 'T1543', 'T1556', 'T1505'],
    'Privilege Escalation': ['T1548', 'T1611', 'T1068', 'T1574', 'T1055'],
    'Defense Evasion': ['T1140', 'T1562', 'T1036', 'T1014', 'T1205'],
    'Credential Access': ['T1110', 'T1555', 'T1606', 'T1111', 'T1040', 'T1539'],
    'Command & Control': ['T1665', 'T1095', 'T1572', 'T1205', 'T1001'],
    'Impact': ['T1531', 'T1485', 'T1486', 'T1561', 'T1491', 'T1495', 'T1657'],
}


def extract_mitre_tactics():
    # Initialize MITRE ATT&CK data
    mitre = MitreAttackData("enterprise-attack.json")

    # Dictionary to store tactics and their techniques
    tactics_dict = {}

    # Get all tactics
    tactics = mitre.get_tactics()

    # Iterate through each tactic
    for tactic in tactics:
        tactic_id = tactic.external_references[0].external_id
        tactic_name = tactic.name

        skip = False
        # Check for conditions
        # Remove tactics
        if tactic_name not in valid_techniques.keys():
            skip = True

        if skip:
            continue

        # Get techniques for this tactic
        techniques = mitre.get_techniques_by_tactic(tactic_name.replace(' ', '-').lower(), 'enterprise-attack',
                                                    remove_revoked_deprecated=True)

        # Store techniques info
        tech_list = []
        for technique in techniques:
            curr_name = technique.name
            curr_id = technique.external_references[0].external_id
            curr_desc = technique.description

            skip = False
            # Check for conditions
            # Remove sub-techniques
            if re.match(r'[\w\d]+\.\d+', curr_id):
                skip = True
            # Remove based on id
            if curr_id not in valid_techniques[tactic_name]:
                skip = True


            curr_desc = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', curr_desc)  # Remove markdown links
            curr_desc = re.sub(r'\(Citation:.*?\)', '', curr_desc)  # Remove citations
            curr_desc = re.sub(r'\s+', ' ', curr_desc).strip()  # Remove extra whitespace

            if skip:
                continue

            tech_info = {
                "technique_id": curr_id,
                "name": curr_name,
                "description": curr_desc
            }
            tech_list.append(tech_info)

        # Store in main dictionary
        tactics_dict[tactic_id] = {
            "name": tactic_name,
            "techniques": tech_list
        }

    # Save to file
    output_file = 'inputs/attacks/cyber-security/tactics.json'
    with open(output_file, 'w') as f:
        json.dump(tactics_dict, f, indent=2)

    print(f'Created {output_file}')
    print(f'tactics_dict keys(): {tactics_dict.keys()}')


if __name__ == '__main__':
    extract_mitre_tactics()
