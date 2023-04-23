import numpy as np

directions = {"center": 0, "up": 1, "right": 2, "down": 3, "left": 4}


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def map_direction(direction_str):
    return directions[direction_str]


def map_move_deltas(direction):
    move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
    if isinstance(direction, int):
        return move_deltas[direction]
    return move_deltas[directions[direction]]


def get_direction_code_from_delta(delta):
    return [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)].index(tuple(delta))


# todo: some memoize on distance computations?
def manhattan_dist_vect_point(loc1, loc2):
    return np.sum(np.abs(loc2 - loc1), 1)


def manhattan_dist_points(loc1, loc2):
    return np.sum(np.abs(loc2 - loc1))


def chebyshev_dist_vect_point(loc1, loc2):
    return np.max(np.abs(loc2 - loc1), 1)


def chebyshev_dist_points(loc1, loc2):
    return np.max(np.abs(loc2 - loc1))


def custom_dist_vect_point(loc1, loc2):
    return np.floor((manhattan_dist_vect_point(loc1, loc2) + chebyshev_dist_vect_point(loc1, loc2)) / 2).astype(int)


def custom_dist_points(loc1, loc2):
    return np.floor((manhattan_dist_points(loc1, loc2) + chebyshev_dist_points(loc1, loc2)) / 2).astype(int)


def manh_dist_to_factory_vect_point(loc1, loc2):
    return np.sum(np.maximum(np.abs(loc2 - loc1) - 1, 0), 1)


def manh_dist_to_factory_points(loc1, loc2):
    return np.sum(np.maximum(np.abs(loc2 - loc1) - 1, 0))


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
# def dir_to(pos_diff):
#     if pos_diff[0] == 0 and pos_diff[1] == 0:
#         return 0
#     if abs(pos_diff[0]) >= abs(pos_diff[1]):
#         if pos_diff[0] > 0:
#             return 2
#         else:
#             return 4
#     else:
#         if pos_diff[1] > 0:
#             return 3
#         else:
#             return 1

# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
# clockwise circulation implementation, i.e. priorities are "up" > "right" > "down" > "left"
def dir_to(pos_diff):
    if pos_diff[0] == 0 and pos_diff[1] == 0:
        return directions["center"]
    if pos_diff[1] < 0:
        return directions["up"]
    elif pos_diff[0] > 0:
        return directions["right"]
    elif pos_diff[1] > 0:
        return directions["down"]
    return directions["left"]


def delete_value_from_dict(d, v):
    for k in list(d.keys()):
        if d[k] == v:
            del d[k]


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
# returns whether_adjacent_to_factory, dir_to_factory
def adjacent_to_factory(pos, fact_pos):
    pos_diff = fact_pos - pos
    abs_diff = np.abs(pos_diff)
    if np.max(abs_diff) <= 2 and np.min(abs_diff) <= 1:
        return True, dir_to(pos_diff)
    return False, 0


# returns the position within the factory that is the nearest to pos
# def nearest_factory_tile(pos, fact_pos):
#     dx, dy = 0, 0
#     if pos[0] > fact_pos[0]:
#         dx += 1
#     elif pos[0] < fact_pos[0]:
#         dx -= 1
#     if pos[1] > fact_pos[1]:
#         dy += 1
#     elif pos[1] < fact_pos[1]:
#         dy -= 1
#     return fact_pos + np.array([dx, dy])


def nearest_factory_tile(pos, fact_pos, excluded_tiles=None):
    excluded_tiles = [tuple(t) for t in excluded_tiles] if excluded_tiles is not None else []
    obs_n = 1
    distances = dict()
    for dx, dy in [(dx, dy) for dx in range(-obs_n, obs_n+1) for dy in range(-obs_n, obs_n+1)]:
        if dx == 0 and dy == 0:
            continue
        t = fact_pos[0] + dx, fact_pos[1] + dy
        if t in excluded_tiles:
            continue
        distances[t] = manhattan_dist_points(np.array(t), pos)
    return np.array(sorted(distances.keys(), key=lambda t: distances[t])[0])


def weighted_rubble_adjacent_density(rubble_board, loc, max_d=7):
    board_size = 48
    weights, rubbles = list(), list()
    for x_i in range(max(loc[0] - max_d, 0), min(loc[0] + max_d, board_size)):
        for x_j in range(max(loc[1] - max_d, 0), min(loc[1] + max_d, board_size)):
            if np.abs(x_i - loc[0]) <= 1 and np.abs(x_j - loc[1]) <= 1:
                continue  # within potential factory location, we skip
            dist = manhattan_dist_points(np.array([x_i, x_j]), loc)
            if dist <= max_d:
                weights.append(max_d + 1 - dist)  # the further, the least important
                rubbles.append(rubble_board[x_i, x_j])
    return np.average(rubbles, weights=weights)


# todo: could be done better, i.e. favoring adjacent areas nearby existing lichen or already dug tiles
# todo: also could optimise perfs doing a base scoring only once per turn per factory, and adding the
#       proximity penalty when sorting positions (in the key arg).
def score_rubble_tiles_to_dig(game_state, associated_factory, obs_n=10, distance_penalty_factor=30):

    # if include_proximity_penalty and unit_pos is None:
    #     raise NotImplementedError("Error in score_rubble_tiles_to_dig: if include_proximity_penalty "
    #                               "then non null unit_pos should be provided")

    tiles_scores = dict()
    for dx, dy in [(dx, dy) for dx in range(-obs_n, obs_n+1) for dy in range(-obs_n+abs(dx), obs_n-abs(dx)+1)]:
        if abs(dx) <= 1 and abs(dy) <= 1:
            continue
        x, y = associated_factory.pos[0] + dx, associated_factory.pos[1] + dy
        if x < 0 or x >= 48 or y < 0 or y >= 48:
            continue
        if game_state.board.factory_occupancy_map[x, y] >= 0:
            continue  # there is a factory in there
        if game_state.board.ice[x, y] or game_state.board.ore[x, y]:
            continue
        rubble = game_state.board.rubble[x, y]
        if not rubble:
            continue

        tile_score = distance_penalty_factor * (min(abs(dx - 1), 0) + min(abs(dy - 1), 0)) + rubble
        tiles_scores[(x, y)] = tile_score

    # if include_proximity_penalty:
    #     tiles_scores = score_rubble_add_proximity_penalty_to_tiles_to_dig(tiles_scores, unit_pos)

    # ordered_tiles = np.array(sorted(tiles_scores.keys(), key=lambda t: tiles_scores[(t[0], t[1])]))
    return tiles_scores


def score_rubble_add_proximity_penalty_to_tiles_to_dig(tiles_scores, unit_pos, proximity_penalty_factor=7):
    return {t: score + proximity_penalty_factor * (abs(unit_pos[0] - t[0]) + abs(unit_pos[1] - t[1]))
            for t, score in tiles_scores.items()}


def count_day_turns(turn_start, turn_end):
    # 30 days then 20 nights, cycles of 50 turns

    day_counter = 0
    cur_turn = turn_start - (turn_start % 50)
    while cur_turn < turn_end:
        day_counter += 30
        if turn_start > cur_turn:
            day_counter -= min((turn_start % 50), 30)
        if turn_end < cur_turn + 30:
            day_counter -= cur_turn + 30 - turn_end
        cur_turn += 50

    return day_counter


def is_registry_free(turn, pos, position_registry, own_unit_id=None):
    # check if registry is free for input. Considered free is booked for own_unit_id
    try:  # not totally sure it's faster than checking if in position_registry.keys(), could try both
        owner_id = position_registry[(turn, tuple(pos))]
        if own_unit_id is not None:
            return owner_id == own_unit_id
    except KeyError:
        return True


def water_cost_to_end(current_cost, turns_remaining, increase_rate=0.04):
    return current_cost * (1 - np.exp(increase_rate) ** (turns_remaining + 1)) / (1 - np.exp(increase_rate))


def assess_units_main_threat(units, op_units):
    """
    for all units, identify the main threat (the nearest unit, biggest if a tie, most power if tie)
    :return: dict my_unit_id: (op_u_id, manh_dist, op_unit.unit_type, op_unit.power)
    """
    return {unit_id: assess_unit_main_threat(unit, op_units) for unit_id, unit in units.items()}


def assess_unit_main_threat(unit, op_units):
    """
    identify the main threat for input tile (the nearest unit, biggest if a tie, most power if tie)
    :return: dict my_unit_id: (op_u_id, manh_dist, op_unit.unit_type, op_unit.power)
    """
    if not len(op_units):
        return None
    dist_to_op_u = manhattan_dist_vect_point(np.array([op_u.pos for op_u in op_units.values()]), unit.pos)
    threats_l = sorted([(op_u_id, dist, op_u.unit_type, op_u.power)
                        for (op_u_id, op_u), dist in zip(op_units.items(), dist_to_op_u)],
                       key=lambda x: (x[1] if x[1] != 1 else 2, -(x[2] == "HEAVY"), x[1], -x[3]))
    return threats_l[0] if len(threats_l) else None


def threat_category(my_unit, their_unit):
    if my_unit.unit_type == "HEAVY" and their_unit.unit_type == "LIGHT":
        return "winning"
    elif my_unit.unit_type == "LIGHT" and their_unit.unit_type == "HEAVY":
        return "losing"
    elif my_unit.power >= their_unit.power:
        return "threaten_winning"
    return "threaten_losing"


def is_unit_stronger(my_unit, their_unit):
    """
    :return: True if my_unit is stronger than their_unit
    """
    if my_unit.unit_type != their_unit.unit_type:
        return my_unit.unit_type == "HEAVY"
    return my_unit.power >= their_unit.power


def is_unit_safe(my_unit, their_unit):
    """
    :return: True if my_unit is much stronger than their unit, i.e. if HEAVY vs LIGHT
    """
    return my_unit.unit_type == "HEAVY" and their_unit.unit_type == "LIGHT"


def get_pos_power_cargo(unit, unit_pos=None, unit_power=None, unit_cargo=None):
    """
    util function to initialise in one line the unit_pos, unit_power and unit_cargo (useful for forward projections)
    :return: unit_pos, unit_power, unit_cargo
    """
    unit_pos = unit.pos if unit_pos is None else unit_pos
    unit_power = unit.power if unit_power is None else unit_power
    unit_cargo = unit.cargo if unit_cargo is None else unit_cargo
    # if unit_cargo is None:
    #     unit_cargo = [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]
    return unit_pos, unit_power, unit_cargo


def factory_lichen_count(factory, game_state):
    return np.sum(game_state.board.lichen[game_state.board.lichen_strains == factory.strain_id])


def approx_factory_lichen_tile_count(factory, game_state, lichen_threshold=50):
    return np.sum([lchn >= lichen_threshold
                   for lchn in game_state.board.lichen[game_state.board.lichen_strains == factory.strain_id]])


# count lichen directly around a factory
def approx_factory_lichen_count(factory, game_state, obs_square_demi_size=4):
    assert obs_square_demi_size >= 2
    lichen_count = 0
    f_x, f_y = factory.pos[0], factory.pos[1]
    for x in range(max(0, f_x-obs_square_demi_size), min(47, f_x+obs_square_demi_size)):
        for y in range(max(0, f_y-obs_square_demi_size), min(47, f_y+obs_square_demi_size)):
            if abs(f_x-x) <= 1 and abs(f_y-x) <= 1:
                continue
            if game_state.board.lichen_strains[x, y] == factory.strain_id:
                lichen_count += game_state.board.lichen[x, y]
    return lichen_count


def find_sweet_attack_spots_on_factory(op_factory, game_state, max_rubble=50, min_lichen=0,
                                       exclude_adj_to_factory_tiles=True):
    f_x, f_y = op_factory.pos[0], op_factory.pos[1]
    all_factories_tiles = np.array([f.pos for f in game_state.factories["player_0"].values()] +
                                   [f.pos for f in game_state.factories["player_1"].values()])

    # define 4 times 3 pairs that can be occupied IF resource here
    # [(occupied_tile), (if_resources_here)]
    resources_t = [[(f_x-3, f_y-1), (f_x-2, f_y-1)], [(f_x-3, f_y), (f_x-2, f_y)], [(f_x-3, f_y+1), (f_x-2, f_y+1)],
                   [(f_x+3, f_y-1), (f_x+2, f_y-1)], [(f_x+3, f_y), (f_x+2, f_y)], [(f_x+3, f_y+1), (f_x+2, f_y+1)],
                   [(f_x-1, f_y-3), (f_x-1, f_y-2)], [(f_x, f_y-3), (f_x, f_y-2)], [(f_x+1, f_y-3), (f_x+1, f_y-3)],
                   [(f_x-1, f_y+3), (f_x-1, f_y+2)], [(f_x, f_y+3), (f_x, f_y+2)], [(f_x+1, f_y+3), (f_x+1, f_y+3)]]

    attack_tiles_resources = list()
    for t_main, t_dep in resources_t:
        if (not 0 <= t_main[0] < 47) or (not 0 <= t_main[1] < 47):
            continue
        if exclude_adj_to_factory_tiles and \
                np.min(manh_dist_to_factory_vect_point(np.array(t_main), all_factories_tiles)) <= 1:
            continue
        if game_state.board.ice[t_dep[0], t_dep[1]] or game_state.board.ice[t_dep[0], t_dep[1]]:
            attack_tiles_resources.append(t_main)

    others_t = [(f_x-3, f_y-1), (f_x-3, f_y+1), (f_x+3, f_y-1), (f_x+3, f_y+1),
                (f_x-1, f_y-3), (f_x+1, f_y-3), (f_x-1, f_y+3), (f_x+1, f_y+3)]
    attack_tiles_lichen = list()
    for t in others_t:
        if (not 0 <= t[0] < 47) or (not 0 <= t[1] < 47):
            continue
        if exclude_adj_to_factory_tiles and \
                np.min(manh_dist_to_factory_vect_point(np.array(t), all_factories_tiles)) <= 1:
            continue
        if game_state.board.lichen[t[0], t[1]] >= min_lichen and \
                ((not game_state.board.lichen[t[0], t[1]]) or
                 game_state.board.lichen_strains[t[0], t[1]] == op_factory.strain_id) \
                and t not in attack_tiles_resources:
            if game_state.board.rubble[t[0], t[1]] <= max_rubble:
                attack_tiles_lichen.append(t)

    return attack_tiles_resources, attack_tiles_lichen


def find_sweet_rest_spots_nearby_factory(factory, game_state):
    f_x, f_y = factory.pos[0], factory.pos[1]
    all_factories_tiles = np.array([f.pos for f in game_state.factories["player_0"].values()] +
                                   [f.pos for f in game_state.factories["player_1"].values()])

    # delta_options = [(6, 3), (6, -3), (-6, 3), (-6, -3),
    #                  (3, 6), (-3, 6), (3, -6), (-3, -6),
    #                  (0, 7), (7, 0), (0, -7), (-7, 0)]
    delta_options = [(6, 0), (0, 6), (-6, 0), (0, -6), (4, 4), (4, -4), (-4, 4), (-4, -4)]

    rest_tiles = list()
    for delta in delta_options:
        new_pos = f_x + delta[0], f_y + delta[1]
        if (not 0 <= new_pos[0] < 47) or (not 0 <= new_pos[1] < 47):
            continue
        if np.min(manh_dist_to_factory_vect_point(np.array(new_pos), all_factories_tiles)) <= 2:
            continue
        if game_state.board.ice[new_pos[0], new_pos[1]] or game_state.board.ice[new_pos[0], new_pos[1]]:
            continue
        rest_tiles.append(new_pos)
    return rest_tiles


def find_guarded_spot(unit, game_state, assigned_factory, bullies_register, dist_to_bullies=5, dist_to_own_factories=6,
                      dist_to_op_factories=6):
    player_me = "player_0" if unit.unit_id in game_state.units["player_0"].keys() else "player_1"
    player_op = "player_1" if player_me == "player_0" else "player_0"
    op_factory_tiles = np.array([f.pos for f in game_state.factories[player_op].values()])
    my_factory_tiles = np.array([f.pos for f in game_state.factories[player_me].values()])
    all_bully_tiles = np.array([t for t in bullies_register.values()])
    # registered_tiles = np.array(op_factory_tiles + my_factory_tiles + all_bully_tiles)
    # op_factory_tiles, my_factory_tiles = np.array()

    # define a reasonable "danger corridor" where to park rather than everywhere
    min_x, min_y = np.min(np.array([t for t in op_factory_tiles]), axis=0)
    min_x, min_y = max(0, min(min_x, assigned_factory.pos[0]) - 1), max(0, min(min_y, assigned_factory.pos[1]) - 1)
    max_x, max_y = np.max(np.array([t for t in op_factory_tiles]), axis=0)
    max_x, max_y = min(47, max(max_x, assigned_factory.pos[0]) + 1), min(47, max(max_y, assigned_factory.pos[1]) + 1)
    if max_x - min_x <= 10:
        min_x, max_x = max(min_x - 3, 0), min(max_x + 3, 47)
    if max_y - min_y <= 10:
        min_y, max_y = max(min_y - 3, 0), min(max_y + 3, 47)

    guarded_tile = None
    n = dist_to_own_factories
    while n <= dist_to_own_factories + 25 and guarded_tile is None:
        for dx in range(-n, n + 1):
            if guarded_tile is not None:
                break
            tile_options = [(dx, n - abs(dx))] if abs(dx) == n else [(dx, n - abs(dx)), (dx, abs(dx) - n)]
            for delta in tile_options:
                t_a = np.array(delta) + assigned_factory.pos
                if (not 0 <= t_a[0] < 48) or (not 0 <= t_a[1] < 48) or (not min_x <= t_a[0] <= max_x) or \
                        (not min_y <= t_a[1] <= max_y):
                    continue  # next move not in the map
                if game_state.board.ice[t_a[0], t_a[1]] or game_state.board.ore[t_a[0], t_a[1]]:
                    continue
                if len(all_bully_tiles) and np.min(manhattan_dist_vect_point(all_bully_tiles, t_a)) < dist_to_bullies:
                    continue
                if len(my_factory_tiles) and \
                        np.min(manhattan_dist_vect_point(my_factory_tiles, t_a)) < dist_to_own_factories:
                    continue
                if len(op_factory_tiles) and \
                        np.min(manhattan_dist_vect_point(op_factory_tiles, t_a)) < dist_to_op_factories:
                    continue
                guarded_tile = t_a  # fine, that's our tile, we go there!
                break
        n += 1
    return tuple(guarded_tile)


def get_tiles_around(pos, radius):
    tiles = list()
    for i in range(radius):
        options = [(i, radius-i), (-i, radius-i), (i, i-radius), (-i, i-radius)]
        for opt in options:
            if (not 0 <= pos[0] + opt[0] < 47) or (not 0 <= pos[1] + opt[1] < 47):
                continue
            tiles.append((pos[0] + opt[0], pos[1] + opt[1]))
    return tiles


def find_intermediary_spot(unit, game_state, target_pos, unit_pos=None, max_dist=20):
    unit_pos = unit.pos if unit_pos is None else unit_pos
    assert manhattan_dist_points(target_pos, unit_pos) >= max_dist

    player_me = "player_0" if unit.unit_id in game_state.units["player_0"].keys() else "player_1"
    player_op = "player_1" if player_me == "player_0" else "player_0"
    my_factory_tiles = np.array([f.pos for f in game_state.factories[player_me].values()])
    op_factory_tiles = np.array([f.pos for f in game_state.factories[player_op].values()])

    for aim_dist in range(max_dist, max(int(max_dist/2), max(max_dist-10, 0)), -1):
        tiles = get_tiles_around(unit_pos, aim_dist)

        for tile in sorted(tiles, key=lambda t: manhattan_dist_points(target_pos, np.array(t))):
            distances_to_factories = chebyshev_dist_vect_point(op_factory_tiles, np.array(tile))
            if len(distances_to_factories) and np.min(distances_to_factories) == 1:
                continue  # don't go through an opponent factory

            distances_to_factories = chebyshev_dist_vect_point(my_factory_tiles, np.array(tile))
            if len(distances_to_factories) and np.min(distances_to_factories) == 1:
                continue  # avoid my factories for that

            if game_state.board.ice[tile[0], tile[1]] or game_state.board.ore[tile[0], tile[1]]:
                continue

            return np.array(tile)

    return None


def prioritise_attack_tiles_nearby(unit, game_state, position_registry, unit_pos=None,
                                   starting_turn=None, lichen_threshold=None, control_area=2):
    if lichen_threshold is None:
        lichen_threshold = 40 if unit.unit_type == "HEAVY" else 10
    init_turn = game_state.real_env_steps if starting_turn is None else starting_turn
    unit_pos = unit.pos if unit_pos is None else unit_pos
    all_factories_tiles = np.array([f.pos for f in game_state.factories["player_0"].values()] +
                                   [f.pos for f in game_state.factories["player_1"].values()])

    op_player = "player_1" if unit.unit_id in game_state.units["player_0"].keys() else "player_0"
    op_strain_ids = [f.strain_id for f in game_state.factories[op_player].values()]
    obs_n = control_area
    attack_options = list()  # [(tile, score)]
    for t in [(dx, dy) for dx in range(-obs_n, obs_n+1) for dy in range(-obs_n+abs(dx), obs_n-abs(dx)+1)]:
        new_pos = unit_pos + np.array(t)
        if (not 0 <= new_pos[0] < 47) or (not 0 <= new_pos[1] < 47):
            continue
        min_dist_to_fact = np.min(manh_dist_to_factory_vect_point(all_factories_tiles, new_pos))
        dist_to_tile = abs(t[0]) + abs(t[1])
        if not min_dist_to_fact:
            continue
        lichen_on_tile = game_state.board.lichen[new_pos[0], new_pos[1]]
        if lichen_on_tile <= lichen_threshold or \
                game_state.board.lichen_strains[new_pos[0], new_pos[1]] not in op_strain_ids:
            continue
        if not all([is_registry_free(init_turn + i + dist_to_tile, new_pos, position_registry, unit.unit_id)
                    for i in range(2)]):
            continue
        score = dist_to_tile * 90 - lichen_on_tile + (130 if min_dist_to_fact == 1 else 0)
        attack_options.append((new_pos, score))

    attack_options = sorted(attack_options, key=lambda opt: opt[1])
    return [a[0] for a in attack_options]


def get_unit_next_action_as_tuple(unit, actions):
    try:
        new_acts = actions[unit.unit_id]
    except KeyError:
        try:
            return tuple(unit.action_queue[0])
        except IndexError:
            return None
    if len(new_acts):
        return tuple(new_acts[0])
    else:
        return None



# def circle_factory_lichen_count(factory, game_state, obs_square_demi_size=4):
#     assert obs_square_demi_size >= 2
#     lichen_count = 0
#     f_x, f_y = factory.pos[0], factory.pos[1]
#     for x in range(max(0, f_x-obs_square_demi_size), min(47, f_x+obs_square_demi_size)):
#         for y in range(max(0, f_y-obs_square_demi_size), min(47, f_y+obs_square_demi_size)):
#             if abs(f_x-x) <= 1 and abs(f_y-x) <= 1:
#                 continue
#
#             if game_state.board.lichen_strains[x, y] == factory.strain_id:
#                 lichen_count += game_state.board.lichen[x, y]
#     return lichen_count
#

# make trivial itinerary between unit and target pos
# favors clock-wise movements, i,e, prioritise up > right > down > left
# very rubble-inefficient, but naturally handles friendly-collisions decently
# def make_itinerary(unit, target_pos, unit_pos=None, rubble=None, units_positions=None, unit_type='HEAVY'):
#     if unit_pos is None:
#         unit_pos = unit.pos  # trick to allow computations from a different spot (forward planning)
#     pos_diff = target_pos - unit_pos
#     actions = list()
#     if pos_diff[0] == 0 and pos_diff[1] == 0:
#         return actions
#     if pos_diff[1] < 0:
#         actions.append(unit.move(directions["up"], repeat=0, n=-pos_diff[1]))
#     if pos_diff[0] > 0:
#         actions.append(unit.move(directions["right"], repeat=0, n=pos_diff[0]))
#     if pos_diff[1] > 0:
#         actions.append(unit.move(directions["down"], repeat=0, n=pos_diff[1]))
#     if pos_diff[0] < 0:
#         actions.append(unit.move(directions["left"], repeat=0, n=-pos_diff[0]))
#     return actions
