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
    return {unit_id: assess_tile_main_threat(unit.pos, op_units) for unit_id, unit in units.items()}


def assess_tile_main_threat(tile, op_units):
    """
    identify the main threat for input tile (the nearest unit, biggest if a tie, most power if tie)
    :return: dict my_unit_id: (op_u_id, manh_dist, op_unit.unit_type, op_unit.power)
    """
    dist_to_op_u = manhattan_dist_vect_point(np.array([op_u.pos for op_u in op_units.values()]), tile)
    threats_l = sorted([(op_u_id, dist, op_u.unit_type, op_u.power)
                        for (op_u_id, op_u), dist in zip(op_units.items(), dist_to_op_u)],
                       key=lambda x: (x[1] if x[1] != 1 else 2, x[2] == "HEAVY", x[1], x[3]))
    return threats_l[0] if len(threats_l) else None


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
