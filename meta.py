import json
import os
from collections import defaultdict


def parse_game_outcome(game_full_log_file):
    with open(game_full_log_file) as f:
        for l in f:
            if "Final Scores:" in l:
                scores = l.split("Final Scores:", 1)[1]
                return eval(scores)
    raise ValueError("unknown outcome")


tmp_file = "tmp.txt"
main_file1 = "main2.py"
main_file2 = "main.py"
nb_of_games = 100

games_stats = defaultdict(int)

for game_i in range(1, nb_of_games + 1):
    seed = game_i
    os.system(f"luxai-s2 {main_file1} {main_file2} > {tmp_file} -v 3 -s {seed}")
    # res = os.system(f"luxai-s2 {main_file1} {main_file2} > {tmp_file} -v 3")
    # with open(tmp_file) as f:
    #     print([l for l in f])

    game_outcome = parse_game_outcome(tmp_file)
    if game_outcome["player_0"] > game_outcome["player_1"]:
        games_stats["player_0"] += 1
    elif game_outcome["player_0"] < game_outcome["player_1"]:
        games_stats["player_1"] += 1
    else:
        games_stats["draw"] += 1
    print(game_i, game_outcome, games_stats)

print(games_stats)
# os.system(f"rm {tmp_file}")

# print(res)

# meh = json.load(open("replay.json"))
# print("done")