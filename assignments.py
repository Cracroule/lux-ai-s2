import collections
import numpy as np

from lux.utils_raoul import chebyshev_dist_points, chebyshev_dist_vect_point, nearest_factory_tile, \
    manh_dist_to_factory_vect_point, manh_dist_to_factory_points, manhattan_dist_points, custom_dist_points, \
    custom_dist_vect_point, score_rubble_tiles_to_dig


def get_ideal_assignment_queue(factory, game_state, assigned_resources, factory_regime,
                               max_distance_to_exploit_ore=10, max_distance_to_exploit_ice=5):
    """
    Ideally, should be function of the turn and the position of the factory...
    Will be improved over time?

    :return: ordered list of assignments
    """

    player_me = "player_0" if factory.unit_id in game_state.units["player_0"].keys() else "player_1"
    player_op = "player_1" if player_me == "player_0" else "player_0"
    tiles_scores = score_rubble_tiles_to_dig(game_state, factory, obs_n=10, op_player=player_op, allow_lichen=False)
    if not len(tiles_scores):
        allow_dig_rubble = False
    else:
        allow_dig_rubble = True

    ice_tiles = np.array(assigned_resources["ice"][factory.unit_id])
    ore_tiles = np.array(assigned_resources["ore"][factory.unit_id])

    # sorted_ice_distances = sorted(custom_dist_vect_point(ice_tiles, factory.pos)) if len(ice_tiles) else []
    sorted_ice_distances = sorted(manh_dist_to_factory_vect_point(ice_tiles, factory.pos)) if len(ice_tiles) else []
    sorted_ore_distances = sorted(manh_dist_to_factory_vect_point(ore_tiles, factory.pos)) if len(ore_tiles) else []
    sorted_ice_tiles = sorted(ice_tiles, key=lambda t: manh_dist_to_factory_points(t, factory.pos))
    sorted_ore_tiles = sorted(ore_tiles, key=lambda t: manh_dist_to_factory_points(t, factory.pos))

    queue = list()

    n_exploited_ore, n_exploited_ice, n_diggers, n_bullies = 0, 0, 0, 0
    n_adjacent_ice = sum([d == 1 for d in sorted_ice_distances])
    n_adjacent_ore = sum([d == 1 for d in sorted_ore_distances])
    max_distance_to_exploit_ore, max_distance_to_exploit_ice = 10, 5

    # TODO: remove below trick
    # n_exploited_ore = 10

    # todo: expend information available: nb of units AND factory details (water available)
    for instruction in factory_regime:
        if instruction == "water":
            if n_adjacent_ice > n_exploited_ice:
                n_exploited_ice += 1
                queue.extend([f"water_duo_main_{n_exploited_ice}", f"water_duo_assist_{n_exploited_ice}"])
            elif len(sorted_ice_distances) > n_exploited_ice \
                    and sorted_ice_distances[n_exploited_ice] <= max_distance_to_exploit_ice:
                n_exploited_ice += 1
                queue.extend([f"water_solo_{n_exploited_ice}"])
        elif instruction == "metal":
            if n_adjacent_ore > n_exploited_ore:
                n_exploited_ore += 1
                queue.extend([f"ore_duo_main_{n_exploited_ore}", f"ore_duo_assist_{n_exploited_ore}"])
            elif len(sorted_ore_distances) > n_exploited_ore \
                    and sorted_ore_distances[n_exploited_ore] <= max_distance_to_exploit_ore:
                n_exploited_ore += 1
                queue.extend([f"ore_solo_{n_exploited_ore}"])
        elif instruction == "rubble":
            if allow_dig_rubble:
                for i in range(2):
                    n_diggers += 1
                    queue.extend([f"dig_rubble_{n_diggers}"])

    for i in range(30):  # then fight x10
        n_bullies += 1
        queue.extend([f"bully_{n_bullies}"])

    # for i in range(10):
    #     n_diggers += 1
    #     queue.extend([f"dig_rubble_{n_diggers}"])
    # below might actually alter the optimal queue
    asngmt_tile_map = assign_tiles(factory, queue, sorted_ice_tiles, sorted_ore_tiles,
                                   max_distance_to_exploit_ore=max_distance_to_exploit_ore,
                                   max_distance_to_exploit_ice=max_distance_to_exploit_ice)
    # check we did the tiles mapping correctly
    assert all([asgnmt in asngmt_tile_map.keys() for asgnmt in queue])

    return queue, asngmt_tile_map


def assign_tiles(factory, queue, sorted_ice_tiles, sorted_ore_tiles, max_distance_to_exploit_ore=10,
                 max_distance_to_exploit_ice=5):
    asngmt_tile_map = dict()
    for asgnmt in queue:
        if "water_" in asgnmt and "_assist_" not in asgnmt:
            resource, n_allocated = asgnmt.split('_')[0], int(asgnmt.split('_')[-1])
            asngmt_tile_map[asgnmt] = sorted_ice_tiles[n_allocated-1]
        elif "ore_" in asgnmt and "_assist_" not in asgnmt:
            resource, n_allocated = asgnmt.split('_')[0], int(asgnmt.split('_')[-1])
            asngmt_tile_map[asgnmt] = sorted_ore_tiles[n_allocated-1]
        elif "_assist_" in asgnmt:
            resource, n_allocated = asgnmt.split('_')[0], int(asgnmt.split('_')[-1])
            if resource == "water" or resource == "ice":
                near_factory_tile_from_dig_spot = nearest_factory_tile(sorted_ice_tiles[n_allocated-1], factory.pos)
            else:
                near_factory_tile_from_dig_spot = nearest_factory_tile(sorted_ore_tiles[n_allocated - 1], factory.pos)
            # todo: find a hack if a tile is meant to be used by two different assistants...
            #       below one should do, i.e. hack task
            if tuple(near_factory_tile_from_dig_spot) not in \
                    [tuple(t) for t in asngmt_tile_map.values() if t is not None]:
                asngmt_tile_map[asgnmt] = near_factory_tile_from_dig_spot
            else:
                new_n = max([int(a.split('_')[-1]) for a in queue] + [1000]) + 1
                new_asgnmt = f"dig_rubble_{new_n}"
                queue[:] = [new_asgnmt if x == asgnmt else x for x in queue]
                asngmt_tile_map[new_asgnmt] = None
            # asngmt_tile_map[asgnmt] = near_factory_tile_from_dig_spot
        else:
            asngmt_tile_map[asgnmt] = None

    return asngmt_tile_map


def update_assignments(factory, factory_units, cur_units_assignments, ideal_queue, heavy_reassign=True,
                       allow_heavy_rubble=False, lim_heavy_per_resource=2):
    """
    look at current assignments, ideal_assignments_queue and reorganise assignments such as it is fit-for-purpose.

    :return: {u_id: reassignment} (only for u_id that have been updated)
    """

    # below useless if we're careful in arg we pass
    cur_units_assignments = {u_id: asgnmt for u_id, asgnmt in cur_units_assignments.items()
                             if u_id in factory_units.keys()}

    # redefine units as only the currently assigned ones...
    # the other ones will be assigned later or are assumed to be assigned already... not great
    # factory_units = {u_id: unit for u_id, unit in factory_units.items() if u_id in cur_factory_assignments.keys()}

    # # assumes no unassigned units... could be improved for that
    # assert (not len(units)) or all([u_id in current_factory_assignments.keys() for u_id in units.keys()])
    # assert all([u_id in cur_factory_assignments.keys() for u_id in factory_units.keys()])
    # unassigned_units_d = {u_id: u for u_id, u in units.items() if u_id not in current_factory_assignments.keys()}

    nb_units = len(factory_units)
    nb_heavy = len([u_id for u_id, unit in factory_units.items() if unit.unit_type == "HEAVY"])
    unassigned_units = [u_id for u_id in factory_units.keys() if u_id not in cur_units_assignments.keys() or
                        cur_units_assignments[u_id] not in ideal_queue]

    # define sorted_u_id that contains all unit_ids sorted by reassignment priority
    priorities_d = {u_id: ideal_queue.index(asgnmt) for u_id, asgnmt in cur_units_assignments.items()
                    if u_id not in unassigned_units}
    u_id_by_desc_priority = sorted(list(priorities_d.keys()), key=lambda u_id: -priorities_d[u_id])
    sorted_u_id = unassigned_units + u_id_by_desc_priority

    # if game_state is not None and game_state.real_env_steps > 100 and factory.unit_id == "factory_1":
    #     pass

    # implem check, to be removed once confident
    assert all([u_id in factory_units.keys() for u_id in sorted_u_id]) and \
           all([u_id in sorted_u_id for u_id in factory_units.keys()])

    reassignments_d = dict()
    if heavy_reassign:
        i = 0
        for asgnmt in ideal_queue:
            if "_assist_" in asgnmt or ((not allow_heavy_rubble) and "dig_rubble" in asgnmt):
                continue

            if lim_heavy_per_resource is not None and asgnmt.startswith("ore_") and \
                    int(asgnmt.split('_')[-1]) > lim_heavy_per_resource:
                continue

            if lim_heavy_per_resource is not None and asgnmt.startswith("water_") and \
                    int(asgnmt.split('_')[-1]) > lim_heavy_per_resource:
                continue

            i += 1
            if i > nb_heavy:
                break

            try:
                currently_assigned_u_id = [u_id for u_id, asgnmt_ in cur_units_assignments.items()
                                           if asgnmt_ == asgnmt][0]
                assigned_unit_type = factory_units[currently_assigned_u_id].unit_type
                if assigned_unit_type == "HEAVY":
                    continue
                # put to the least attractive task for now, will be reassigned properly later on
                # current_factory_assignments[currently_assigned_u_id] = ideal_queue[-1]
                # reassignments_d[currently_assigned_u_id] = ideal_queue[-1]

                sorted_u_id.remove(currently_assigned_u_id)  # remove from wherever it is
                sorted_u_id.insert(0, currently_assigned_u_id)  # put on left side (to be reassigned asap)

            except IndexError:
                # no currently_assigned_u_id to that task, let's take one
                pass

            to_be_reassigned_id = [u_id for u_id in sorted_u_id if factory_units[u_id].unit_type == "HEAVY"][0]
            reassignments_d[to_be_reassigned_id] = asgnmt
            sorted_u_id.remove(to_be_reassigned_id)  # remove from left side
            sorted_u_id.append(to_be_reassigned_id)  # put on right side  (not to be reassigned)

    # define (again!) sorted_u_id that contains all unit_ids sorted by reassignment priority
    # re_cur_units_assignments = dict(cur_units_assignments, **reassignments_d)
    re_cur_units_assignments = {**{u_id: asgnmt for u_id, asgnmt in cur_units_assignments.items()
                                   if asgnmt not in reassignments_d.values()}, **reassignments_d}
    re_unassigned_units = [u_id for u_id in factory_units.keys() if u_id not in re_cur_units_assignments.keys() or
                           re_cur_units_assignments[u_id] not in ideal_queue]
    priorities_d = {u_id: ideal_queue.index(asgnmt) for u_id, asgnmt in re_cur_units_assignments.items()
                    if u_id not in re_unassigned_units}
    u_id_by_desc_priority = sorted(list(priorities_d.keys()), key=lambda u_id: -priorities_d[u_id])
    sorted_u_id = re_unassigned_units + u_id_by_desc_priority

    i = 0
    for asgnmt in ideal_queue:

        if asgnmt in re_cur_units_assignments.values() and \
                factory_units[[u_id for u_id, asgnmt_ in re_cur_units_assignments.items()
                               if asgnmt_ == asgnmt][0]].unit_type == "HEAVY":
            continue

        i += 1
        if i > nb_units - (nb_heavy if heavy_reassign else 0):
            break

        if asgnmt not in re_cur_units_assignments.values():
            # reassign unit with the smallest priority to something more important

            j = 0
            if heavy_reassign:
                while factory_units[sorted_u_id[j]].unit_type == "HEAVY" and j < nb_units:
                    j += 1
                if j == nb_units:
                    break  # we're done with reassignment
            to_be_reassigned_id = sorted_u_id[j]

            # to_be_reassigned_id = sorted_u_id[0]
            # if factory_units[to_be_reassigned_id].unit_type == "HEAVY":
            #     break  # do not reassign an HEAVY one. We're done if that's the only option (only assist filling)

            reassignments_d[to_be_reassigned_id] = asgnmt
            sorted_u_id.remove(to_be_reassigned_id)  # remove from left side
            sorted_u_id.append(to_be_reassigned_id)  # put on right side  (not to be reassigned)

    # implem check, to be removed once confident
    assert all([u_id in factory_units.keys() for u_id in sorted_u_id]) and \
           all([u_id in sorted_u_id for u_id in factory_units.keys()])

    return reassignments_d


# def give_assignment(unit, game_state, robots_assignments, map_unit_factory, assigned_resources, ideal_queue,
#                     assigned_factory=None):
#     """
#     Create an assignment based on known info, i.e. already assigned robots and map robot:factory.
#     For now, assignments come in a specific order
#     ideally, any robot who does not know what to do should consult this assignment and act accordingly
#     ideally, they would consult it in order of strength, i.e. bored heavy robots first, then bored light ones
#
#     :param unit: not used for now. Could be used to know if light/heavy later on, or based on current position
#     :param game_state: not used for now, could be used to get info on board
#     :param robots_assignments: dictionary robot_id: assignment
#     :param map_unit_factory: dictionary unit_id: assigned_factory_id
#     :param assigned_factory: desired factory the input robot should be associated with
#     :return: assignment
#     """
#
#     # # should ideally do the computation of map_factory_units only once per turn
#     # map_factory_units = collections.defaultdict(list)
#     # for u_id, f_id in map_unit_factory.items():
#     #     map_factory_units[f_id].append(u_id)
#
#     # assigned_factory = map_unit_factory[unit.unit_id]
#
#     if assigned_factory is None:
#         # todo: assign factory based on game_state...
#         raise NotImplementedError("assigned_factory must be used for now to get an assignment...")
#     # if ideal_queue is None:
#     #     ideal_queue, _ = get_ideal_assignment_queue(agent, assigned_factory, game_state,
#     #                                                 assigned_resources=assigned_resources)
#
#     # existing_factory_assignments = [robots_assignments[u_id] for u_id in map_factory_units[assigned_factory.unit_id]
#     #                                 if u_id in robots_assignments.keys()]
#
#     chosen_asgnmt = None
#     for asgnmt in ideal_queue:
#         if asgnmt not in existing_factory_assignments:
#             chosen_asgnmt = asgnmt
#             break
#     if chosen_asgnmt is None:
#         chosen_asgnmt = "fight"
#
#     return chosen_asgnmt


def assign_factories_resource_tiles(factories, game_state):
    ice_tile_locations = np.argwhere(game_state.board.ice == 1)
    ore_tile_locations = np.argwhere(game_state.board.ore == 1)

    factory_tiles = np.array([factory.pos for factory in factories.values()])
    factory_ids = list(factories.keys())

    map_factory_tiles = {"ice": collections.defaultdict(list), "ore": collections.defaultdict(list)}
    for tile in ice_tile_locations:
        # factory_distances = chebyshev_dist_vect_point(factory_tiles, tile)
        factory_distances = manh_dist_to_factory_vect_point(factory_tiles, tile)
        near_factory_i = np.argmin(factory_distances)
        map_factory_tiles["ice"][factory_ids[near_factory_i]].append((tile[0], tile[1]))
    for tile in ore_tile_locations:
        # factory_distances = chebyshev_dist_vect_point(factory_tiles, tile)
        factory_distances = manh_dist_to_factory_vect_point(factory_tiles, tile)
        near_factory_i = np.argmin(factory_distances)
        map_factory_tiles["ore"][factory_ids[near_factory_i]].append((tile[0], tile[1]))

    return map_factory_tiles


def decide_factory_regime(factory, game_state, factory_units, assigned_resources, max_distance_to_exploit_ore=10,
                          max_distance_to_exploit_ice=5):

    ice_tiles = np.array(assigned_resources["ice"][factory.unit_id])
    ore_tiles = np.array(assigned_resources["ore"][factory.unit_id])
    sorted_ice_distances = sorted(manh_dist_to_factory_vect_point(ice_tiles, factory.pos)) if len(ice_tiles) else []
    sorted_ore_distances = sorted(manh_dist_to_factory_vect_point(ore_tiles, factory.pos)) if len(ore_tiles) else []

    n_exploitable_ice = sum([d for d in sorted_ice_distances if d <= max_distance_to_exploit_ice])
    n_exploitable_ore = sum([d for d in sorted_ore_distances if d <= max_distance_to_exploit_ore])

    nb_units = len(factory_units)
    nb_heavy = len([u_id for u_id, unit in factory_units.items() if unit.unit_type == "HEAVY"])
    f_power, f_water, f_metal = factory.power, factory.cargo.water, factory.cargo.metal
    cur_turn = game_state.real_env_steps

    # if f_water < 160:
    #     regime = ["water", "rubble", "metal", "water", "rubble", "metal", "rubble", "rubble", "water", "rubble"]
    # elif nb_units >= 3 and f_power < 600:
        #     if f_metal < 400:
        #         regime = ["water", "rubble", "metal", "water", "rubble", "metal"]
        #     else:
        #         regime = ["water", "rubble", "water", "rubble"]
    # elif nb_units < 3:
    #     if f_water > 300:
    #         regime = ["metal", "water", "rubble", "metal", "water", "rubble"]
    #     else:
    #         regime = ["water", "metal", "rubble", "water", "metal", "rubble"]
    # elif f_power > 1500:
    #     regime = ["metal", "water", "rubble", "metal", "water", "rubble"]
    # elif cur_turn < 900 and f_metal < 110:
    #     regime = ["metal", "water", "metal", "water", "rubble", "rubble"]
    # elif cur_turn < 900 and f_metal >= 110:
    #     regime = ["water", "metal", "water", "metal", "rubble", "rubble"]
    # else:  # no time to get metal anymore
    #     regime = ["water", "water", "rubble", "rubble", "metal", "metal"]

    if f_water < 160:
        regime = ["water", "rubble", "metal", "water", "rubble", "metal"]
    elif nb_units >= 3 and f_power < 600:
        # regime = ["water", "rubble", "metal", "water", "rubble", "metal"]
        if f_metal < 400:
            regime = ["water", "rubble", "metal", "water", "rubble", "metal"]
        else:
            regime = ["water", "rubble", "water", "rubble"]
    elif nb_units < 3:
        # regime = ["water", "metal", "rubble", "water", "metal", "rubble"]
        if f_water > 300:
            regime = ["metal", "water", "rubble", "metal", "water", "rubble"]
        else:
            regime = ["water", "metal", "rubble", "water", "metal", "rubble"]
    elif f_power > 1500:
        regime = ["metal", "water", "rubble", "metal", "water", "rubble"]
    elif cur_turn < 900 and f_metal < 110:
        regime = ["metal", "water", "metal", "water", "rubble", "rubble"]
    elif cur_turn < 900 and f_metal >= 110:
        regime = ["water", "metal", "water", "metal", "rubble", "rubble"]
    else:  # no time to get metal anymore
        regime = ["water", "water", "rubble", "rubble", "metal", "metal"]

    # return [*regime, "rubble", *regime]
    return [*regime, *regime]
