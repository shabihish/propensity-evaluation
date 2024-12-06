import autogen

import inspect


def get_classes(module):
    classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            classes.append(obj)
    return classes


def main():
    config_list = autogen.config_list_from_json(
        env_or_file='OAI_CONFIG_LIST.json',
        filter_dict={
            "model": ["gemini-1.5-flash"],
        }
    )

    llm_config = {
        "cache_seed": 41,
        "config_list": config_list,
        "temperature": 0,
        "timeout": 120,  # seconds
    }

    agent = autogen.AssistantAgent(
        name="planner",
        llm_config=llm_config
    )

    user_proxy = autogen.UserProxyAgent(
        name="user",
        human_input_mode='TERMINATE',
        code_execution_config={
            "work_dir": "agent_coding",
            "use_docker": False,
        }
    )

    user_proxy.initiate_chat(agent,
                             message="Plan a "
                             )


if __name__ == "__main__":
    main()
