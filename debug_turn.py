import json
from typing import Dict
import sys
from argparse import Namespace
import joblib

# from agent_rules_based_tmp2 import Agent
# from agent import Agent
# from agent_rules_based_tmp2 import Agent
from agent_new_era import Agent
from lux.config import EnvConfig
from lux.kit import GameState, process_obs, to_json, from_json, process_action, obs_to_game_state

# DO NOT REMOVE THE FOLLOWING CODE #
agent_dict = dict()  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()


def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step

    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        env_cfg = EnvConfig.from_dict(configurations["env_cfg"])
        agent_dict[player] = Agent(player, env_cfg)
        agent_prev_obs[player] = dict()
        agent = agent_dict[player]
    agent = agent_dict[player]
    obs = process_obs(player, agent_prev_obs[player], step, json.loads(observation.obs))
    agent_prev_obs[player] = obs
    agent.step = step
    if obs["real_env_steps"] < 0:
        actions = agent.early_setup(step, obs, remainingOverageTime)
    else:
        actions = agent.act(step, obs, remainingOverageTime)

    return process_action(actions)


if __name__ == "__main__":

    # step = 0
    # player_id = 0
    # configurations = None
    # i = 0

    # always run first turn to initialise set up
    # observation, d = joblib.load(f"data/export_1.joblib")
    # agent_fn(observation, d)

    for i in range(1, 1000):
        print("debug: ", i)
        # then replay whatever turn with possibility of debugging
        export_file = f"data/export_{i}.joblib"
        observation, d = joblib.load(export_file)
        if i == 17:
            print("debug time!")
        actions = agent_fn(observation, d)
    # print()
    # send actions to engine
    print(json.dumps(actions))