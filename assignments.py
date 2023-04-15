import collections
import numpy as np

from lux.utils_raoul import chebyshev_distance_points, chebyshev_dist_vect_point, nearest_factory_tile, \
    custom_dist_vect_point, custom_dist_points, manhattan_dist_points


def get_ideal_assignment_queue(factory, game_state, assigned_resources):
    """
    Ideally, should be function of the turn and the position of the factory...
    Will be improved over time?

    :return: ordered list of assignments
    """

    # todo: prioritise tiles with manhattan distance, we're using the custom one only because convenient for adjacent

    # todo: reorder in a more simple way ?

    # todo: can't have two tasks assigned to the same tile in case of assistants.
    #       If that's the case, should cancel the need for an assistant and replace with solo mode

    # here, we should decide if we go for water with an assistant, or just for water
    # not sure if it supports 2 units with same assignment, should add an end-number to indicate priority

    ice_tiles = np.array(assigned_resources["ice"][factory.unit_id])
    ore_tiles = np.array(assigned_resources["ore"][factory.unit_id])

    # todo: replace the custom_dist by the manh_dist_to_facrory (not to the center of the factory)
    sorted_ice_distances = sorted(custom_dist_vect_point(ice_tiles, factory.pos)) if len(ice_tiles) else []
    sorted_ore_distances = sorted(custom_dist_vect_point(ore_tiles, factory.pos)) if len(ore_tiles) else []
    sorted_ice_tiles = sorted(ice_tiles, key=lambda t: custom_dist_points(t, factory.pos))
    sorted_ore_tiles = sorted(ore_tiles, key=lambda t: custom_dist_points(t, factory.pos))
    # sorted_ice_distances, sorted_ore_distances = sorted(ice_distances), sorted(ore_distances)

    # ice_dist = sorted(set(manhattan_dist_vect_point(ice_tile_locations, loc)))
    # nearest_ice_dist, second_nearest_ice_dist = ice_dist[0], ice_dist[1]

    # ideal_queue = ["water_1_main", "water_1_assist", "ore1", "water2", "dig1", "dig2", "dig3", "dig4", "fight"]
    # ideal_queue = ["water_duo_main_1", "water_duo_assist_1", "ore1", "water2", "dig1", "dig2", "dig3", "dig4", "fight"]
    # ["water_duo_main_1", "ore_duo_main_1", "water_duo_assist_1", "ore_duo_assist_1"] or "water_solo_1" / "ore_solo_1"

    queue = list()
    asngmt_tile_map = dict()

    # first water/ore handling:
    n_exploited_ore, n_exploited_ice = 0, 0
    if len(sorted_ice_distances) and sorted_ice_distances[0] == 2 and \
            len(sorted_ore_distances) and sorted_ore_distances[0] == 2:
        queue.extend(["water_duo_main_1", "ore_duo_main_1", "water_duo_assist_1", "ore_duo_assist_1"])
        # n_exploited_ore += 1
    elif len(sorted_ice_distances) and sorted_ice_distances[0] == 2:
        queue.extend(["water_duo_main_1"])
        if len(sorted_ore_distances) and sorted_ore_distances[0] <= 10:
            queue.extend(["ore_solo_1"])
            # n_exploited_ore += 1
        queue.extend(["water_duo_assist_1"])
    elif len(sorted_ice_distances):
        queue.extend(["water_solo_1"])
        if len(sorted_ore_distances) and sorted_ore_distances[0] <= 10:
            queue.extend(["ore_solo_1"])
            # n_exploited_ore += 1
    else:  # no water :'(  # todo: in that case, spend all ore and sacrifice factory!
        if len(sorted_ore_distances) and sorted_ore_distances[0] <= 10:
            queue.extend(["ore_solo_1"])
            # n_exploited_ore += 1

    # handle asngmt_tile_map for above first water/ore
    if "water_duo_main_1" in queue:
        # near_ice_i = ice_distances.index(sorted_ice_distances[n_exploited_ice])
        # asngmt_tile_map["water_duo_main_1"] = ice_tiles[near_ice_i]
        asngmt_tile_map["water_duo_main_1"] = sorted_ice_tiles[n_exploited_ice]
        near_factory_tile_from_dig_spot = nearest_factory_tile(sorted_ice_tiles[n_exploited_ice], factory.pos)
        asngmt_tile_map["water_duo_assist_1"] = near_factory_tile_from_dig_spot
        n_exploited_ice += 1
    elif "water_solo_1" in queue:
        # near_ice_i = ice_distances.index(sorted_ice_distances[n_exploited_ice])
        # asngmt_tile_map["water_solo_1"] = ice_tiles[near_ice_i]
        asngmt_tile_map["water_solo_1"] = sorted_ice_tiles[n_exploited_ice]
        n_exploited_ice += 1
    if "ore_duo_main_1" in queue:
        # near_ore_i = ore_distances.index(sorted_ore_distances[n_exploited_ore])
        # asngmt_tile_map["ore_duo_main_1"] = ore_tiles[near_ore_i]
        asngmt_tile_map["ore_duo_main_1"] = sorted_ore_tiles[n_exploited_ore]
        near_factory_tile_from_dig_spot = nearest_factory_tile(sorted_ore_tiles[n_exploited_ore], factory.pos)
        asngmt_tile_map["ore_duo_assist_1"] = near_factory_tile_from_dig_spot
        n_exploited_ore += 1
    elif "ore_solo_1" in queue:
        # near_ore_i = ore_distances.index(sorted_ore_distances[n_exploited_ore])
        # asngmt_tile_map["ore_solo_1"] = ore_tiles[near_ore_i]
        asngmt_tile_map["ore_solo_1"] = sorted_ore_tiles[n_exploited_ore]
        n_exploited_ore += 1

    # dig handling
    queue.extend(["dig_rubble_1", "dig_rubble_2"])
    asngmt_tile_map["dig_rubble_1"], asngmt_tile_map["dig_rubble_2"] = None, None  # dig will be organised differently

    # second water handling:
    if len(sorted_ice_distances) > 1 and sorted_ice_distances[1] == 2:
        queue.extend(["water_duo_main_2", "water_duo_assist_2"])  # adjacent water
        # near_ice_i = sorted_ice_distances.index(sorted_ice_distances[n_exploited_ice])
        # asngmt_tile_map["water_duo_main_2"] = ice_tiles[near_ice_i]
        asngmt_tile_map["water_duo_main_2"] = sorted_ice_tiles[n_exploited_ice]
        near_factory_tile_from_dig_spot = nearest_factory_tile(sorted_ice_tiles[n_exploited_ice], factory.pos)
        asngmt_tile_map["water_duo_assist_2"] = near_factory_tile_from_dig_spot
        n_exploited_ice += 1
    elif len(sorted_ice_distances) > 1 and sorted_ice_distances[1] <= 4:
        queue.extend(["water_solo_2"])  # water a bit far
        # near_ice_i = ice_distances.index(sorted_ice_distances[n_exploited_ice])
        # asngmt_tile_map["water_solo_2"] = ice_tiles[near_ice_i]
        asngmt_tile_map["water_solo_2"] = sorted_ice_tiles[n_exploited_ice]
        n_exploited_ice += 1

    # dig handling
    queue.extend(["dig_rubble_3"])
    asngmt_tile_map["dig_rubble_3"] = None  # dig will be handled differently

    # second ore handling:
    if len(sorted_ore_distances) >= (n_exploited_ore + 1) and sorted_ore_distances[n_exploited_ore] <= 10:
        queue.extend([f"ore_solo_{n_exploited_ore + 1}"])
        # near_ore_i = ore_distances.index(sorted_ore_distances[n_exploited_ore])
        # asngmt_tile_map[f"ore_solo_{n_exploited_ore + 1}"] = ore_tiles[near_ore_i]
        asngmt_tile_map[f"ore_solo_{n_exploited_ore + 1}"] = sorted_ore_tiles[n_exploited_ore]
        n_exploited_ore += 1

    # dig more
    queue.extend(["dig_rubble_4"])
    asngmt_tile_map["dig_rubble_4"] = None  # dig will be handled differently

    # fill with fight for now
    for i in range(10):
        queue.extend([f"fight_{i}"])
        asngmt_tile_map[f"fight_{i}"] = None  # fight will be handled differently

    # check we did the tiles mapping correctly
    assert all([asgnmt in asngmt_tile_map.keys() for asgnmt in queue])

    return queue, asngmt_tile_map


def update_assignments(current_factory_assignments, ideal_queue, units, heavy_reassign=True):
    """
    look at current assignments, ideal_assignments_queue and reorganise assignments such as it is fit-for-purpose.

    :return: {u_id: reassignment} (only for u_id that have been updated)
    """

    # redefine units as only the currently assigned ones...
    # the other ones will be assigned later or are assumed to be assigned already... not great
    units = {u_id: unit for u_id, unit in units.items() if u_id in current_factory_assignments.keys()}

    # # assumes no unassigned units... could be improved for that
    # assert (not len(units)) or all([u_id in current_factory_assignments.keys() for u_id in units.keys()])
    assert all([u_id in current_factory_assignments.keys() for u_id in units.keys()])
    # unassigned_units_d = {u_id: u for u_id, u in units.items() if u_id not in current_factory_assignments.keys()}

    nb_heavy = len([u_id for u_id, unit in units.items() if unit.unit_type == "HEAVY"])

    for u_id, asgnmt in current_factory_assignments.items():
        if asgnmt not in ideal_queue:
            print("aieaieaie")

    priorities_d = {u_id: ideal_queue.index(asgnmt) for u_id, asgnmt in current_factory_assignments.items()}
    u_id_by_priority = sorted(list(priorities_d.keys()), key=lambda u_id: priorities_d[u_id])

    reassignments_d = dict()

    if heavy_reassign:
        i = 0
        for asgnmt in ideal_queue:
            if "assist" in asgnmt:
                continue

            i += 1
            if i > nb_heavy:
                break

            try:
                assigned_unit_id = [u_id for u_id, asgnmt_ in current_factory_assignments.items()][0]
                assigned_unit_type = units[assigned_unit_id].unit_type
                if assigned_unit_type == "HEAVY":
                    continue
                # put to the least attractive task for now, will be reassigned properly later on
                # current_factory_assignments[assigned_unit_id] = ideal_queue[-1]
                reassignments_d[assigned_unit_id] = ideal_queue[-1]
            except IndexError:
                # no assigned_unit_id to that task, let's take one
                pass

            to_be_reassigned_id = [u_id for u_id in u_id_by_priority[::-1] if units[u_id].unit_type == "HEAVY"][0]
            reassignments_d[to_be_reassigned_id] = asgnmt
            u_id_by_priority.remove(to_be_reassigned_id)

        # reupdate priorities to include changes
        priorities_d = {u_id: ideal_queue.index(asgnmt) for u_id, asgnmt in dict(current_factory_assignments,
                                                                                 **reassignments_d).items()}
        u_id_by_priority = sorted(list(priorities_d.keys()), key=lambda u_id: priorities_d[u_id])

    i = 1
    for asgnmt in ideal_queue:
        if i > len(units):
            break
        if asgnmt not in dict(current_factory_assignments, **reassignments_d).values():
            # reassign unit with the smallest priority to something more important
            reassignments_d[u_id_by_priority[-1]] = asgnmt
            del u_id_by_priority[-1]
        i += 1

    return reassignments_d


def give_assignment(unit, game_state, robots_assignments, map_unit_factory, assigned_resources, ideal_queue,
                    assigned_factory=None):
    """
    Create an assignment based on known info, i.e. already assigned robots and map robot:factory.
    For now, assignments come in a specific order
    ideally, any robot who does not know what to do should consult this assignment and act accordingly
    ideally, they would consult it in order of strength, i.e. bored heavy robots first, then bored light ones

    :param unit: not used for now. Could be used to know if light/heavy later on, or based on current position
    :param game_state: not used for now, could be used to get info on board
    :param robots_assignments: dictionary robot_id: assignment
    :param map_unit_factory: dictionary unit_id: assigned_factory_id
    :param assigned_factory: desired factory the input robot should be associated with
    :return: assignment
    """

    # should ideally do the computation of map_factory_units only once per turn
    map_factory_units = collections.defaultdict(list)
    for u_id, f_id in map_unit_factory.items():
        map_factory_units[f_id].append(u_id)

    # assigned_factory = map_unit_factory[unit.unit_id]

    if assigned_factory is None:
        # todo: assign factory based on game_state...
        raise NotImplementedError("assigned_factory must be used for now to get an assignment...")
    # if ideal_queue is None:
    #     ideal_queue, _ = get_ideal_assignment_queue(agent, assigned_factory, game_state,
    #                                                 assigned_resources=assigned_resources)

    existing_factory_assignments = [robots_assignments[u_id] for u_id in map_factory_units[assigned_factory.unit_id]
                                    if u_id in robots_assignments.keys()]
    chosen_asgnmt = None
    for asgnmt in ideal_queue:
        if asgnmt not in existing_factory_assignments:
            chosen_asgnmt = asgnmt
            break
    if chosen_asgnmt is None:
        chosen_asgnmt = "fight"

    return chosen_asgnmt


def assign_factories_resource_tiles(factories, game_state):
    ice_tile_locations = np.argwhere(game_state.board.ice == 1)
    ore_tile_locations = np.argwhere(game_state.board.ore == 1)

    factory_tiles = np.array([factory.pos for factory in factories.values()])
    factory_ids = list(factories.keys())

    map_factory_tiles = {"ice": collections.defaultdict(list), "ore": collections.defaultdict(list)}
    for ice_tile in ice_tile_locations:
        factory_distances = chebyshev_dist_vect_point(factory_tiles, ice_tile)
        near_factory_i = np.argmin(factory_distances)
        map_factory_tiles["ice"][factory_ids[near_factory_i]].append((ice_tile[0], ice_tile[1]))
    for ice_tile in ore_tile_locations:
        factory_distances = chebyshev_dist_vect_point(factory_tiles, ice_tile)
        near_factory_i = np.argmin(factory_distances)
        map_factory_tiles["ore"][factory_ids[near_factory_i]].append((ice_tile[0], ice_tile[1]))

    return map_factory_tiles
