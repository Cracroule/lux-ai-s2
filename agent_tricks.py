import collections
import numpy as np
import random
import time
import scipy
import sys
from lux.utils_raoul import get_direction_code_from_delta, manhattan_dist_vect_point, manhattan_dist_points, \
    adjacent_to_factory, nearest_factory_tile, chebyshev_dist_vect_point, chebyshev_dist_points, \
    score_rubble_tiles_to_dig, score_rubble_add_proximity_penalty_to_tiles_to_dig, custom_dist_points, \
    is_registry_free, count_day_turns, is_unit_stronger, get_pos_power_cargo, manh_dist_to_factory_vect_point, \
    manh_dist_to_factory_points, find_sweet_attack_spots_on_factory, find_guarded_spot, find_intermediary_spot, \
    find_sweet_rest_spots_nearby_factory, prioritise_attack_tiles_nearby, threat_category, delete_value_from_dict
# from scipy.spatial.distance import cdist

from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory


# debug / analysis purposes
monitored_units = []
monitored_turns = []
time_monitored_turns = []


# TODO: FOR ALL (Especially solo diggers): IF LOW ENERGY, CHARGE (ELSE CAN END UP BEING STUCK ON LOW ENERGY...)
# TODO: COLLISIONS BECAUSE OF POWER/DIG STEALING THE TILE SOMEONE COUNTED ON
# TODO: DETECT ENEMY, MOVE DIFFERENTLY


def go_adj_dig(unit, game_state, position_registry, target_dug_tile, assigned_factory, factories_power, n_min_digs=2,
               n_desired_digs=5):
    """
    - [pick power]
    - go to tile
    - dig, (repeat)
    - transfer, (repeat)

    # important to start with power to prevent over picking power which makes game crash)
    (assumes will be stopped if running low in power, in which case needs to refill if it does not have an assistant)
    :return: [actions]
    """

    is_ice_dig = game_state.board.ice[target_dug_tile[0], target_dug_tile[1]]
    is_ore_dig = game_state.board.ore[target_dug_tile[0], target_dug_tile[1]]
    assert is_ice_dig or is_ore_dig

    # make sure we're only dealing with adjacent-to-factory dug tiles
    near_factory_tile_from_dig_spot = nearest_factory_tile(target_dug_tile, assigned_factory.pos)
    assert manhattan_dist_points(near_factory_tile_from_dig_spot, target_dug_tile) == 1

    actions, actions_counter = list(), 0
    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit)

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    is_adj, dir_to_fac = adjacent_to_factory(unit_pos, assigned_factory.pos)
    if is_adj:  # if nearby factory + carrying resources: transfer them (reassignment handling) and stop (power after)
        cur_cargo = [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]
        for cargo_amt, r_code in cur_cargo:
            if cargo_amt and is_registry_free(init_turn+actions_counter+1, unit_pos, position_registry, unit.unit_id):
                actions.extend([unit.transfer(dir_to_fac, r_code, cargo_amt)])
                actions_counter += 1
                position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})
                return actions

    dist = manhattan_dist_points(unit_pos, target_dug_tile)
    dist2 = manhattan_dist_points(unit_pos, assigned_factory.pos)
    nb_rubble = 60  # proxy rubble for simplicity, should check actual rubble on map
    power_cost_move = int((dist + dist2) * (unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * nb_rubble))
    desired_buffer = 0

    # pick power if necessary
    res = take_power(unit, game_state, position_registry, assigned_factory, factories_power, power_cost_move, unit_pos,
                     unit_power, unit_cargo, starting_turn=init_turn+actions_counter, n_min_digs=n_min_digs,
                     n_desired_digs=n_desired_digs, desired_buffer=desired_buffer, slow_if_low_power=False)

    if res["stop_after"]:
        return res["actions"]

    # apply res to current context
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit, res["unit_pos"], res["unit_power"], res["unit_cargo"])
    actions.extend(res["actions"])
    position_registry.update(res["position_registry"])
    actions_counter += res["actions_counter"]

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    # go to tile
    # # todo: make itinerary computation before power (but consequences after), so we can use exact power computation
    # #       instead of proxy
    itin_d = make_itinerary_advanced(unit, target_dug_tile, game_state, position_registry,
                                     starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
    actions.extend(itin_d["actions"])
    actions_counter += itin_d["actions_counter"]
    position_registry.update(itin_d["position_registry"])
    unit_pos, power_cost = itin_d["unit_pos"], itin_d["power_cost"]

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    if not np.array_equal(unit_pos, target_dug_tile):
        return actions  # could not find a way, let's wait...

    # if any rubble, dig until no more ruble
    n_dig_rubble = int(np.ceil(game_state.board.rubble[target_dug_tile[0], target_dug_tile[1]] /
                               unit_cfg.DIG_RESOURCE_GAIN))
    if n_dig_rubble:
        actions.extend([unit.dig(repeat=0, n=n_dig_rubble)])
        for i in range(n_dig_rubble):
            actions_counter += 1
            position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    # todo: MASSIVE ISSUE: all the position_registry.update that are NOT following a make_itinerary are actually
    #       not checking for availability of the tile, they just take. They should be able to compromise if
    #       another robot intends to use the tile...
    #   two options:
    #     -implement booking space (with potential move around waiting for spot to be free, or to find alternative spot)
    #     OR    - implement collision prevention, i.e. stop least urgent before collision

    # theoretically from here, the unit will keep digging and transferring forever, thus not moving unless interrupted
    for turn in range(init_turn + actions_counter + 1, 1000 + 1):
        position_registry[(turn, tuple(unit_pos))] = unit.unit_id
        # position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    # dig
    # todo: actually book the digging place... but also means handling non availability
    actions.extend([unit.dig(repeat=n_desired_digs, n=n_desired_digs)])
    actions_counter += n_desired_digs
    # for i in range(n_desired_digs):
    #     actions_counter += 1
    #     position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    # transfer resources
    resource_type = 0 if is_ice_dig else 1
    transfer_direction = direction_to(target_dug_tile, near_factory_tile_from_dig_spot)
    actions.extend([unit.transfer(transfer_direction, resource_type,
                                  n_desired_digs * unit_cfg.DIG_RESOURCE_GAIN, repeat=1)])
    # actions_counter += 1
    # position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})
    return actions


def assist_adj_digging(unit, game_state, position_registry, assisted_unit, assigned_factory, factories_power,
                       target_dug_tile=None, n_desired_digs=5):
    """
    - [pick power] (48 for heavy assisted_unit) (repeat)
    - go to the nearest tile within factory
    - transfer power to assisted_unit (repeat)

    # important to start by pick to prevent over picking power which makes game crash)

    :return: [actions]
    """

    actions, actions_counter = list(), 0
    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    assist_unit_cfg = game_state.env_cfg.ROBOTS[assisted_unit.unit_type]
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit)

    # ignore below for now
    # # then the unit is not on the factory! we bring it back to the factory and that's it, can't take power in the future
    # if manhattan_dist_points(target_factory.pos, unit.pos) > 1:
    #     return actions.extend(go_back_to_factory(unit, game_state, agent, target_factory))

    if target_dug_tile is None:
        target_dug_tile = assisted_unit.pos

    # make sure we're only dealing with adjacent-to-factory dug tiles
    near_factory_tile_from_dig_spot = nearest_factory_tile(target_dug_tile, assigned_factory.pos)
    assert manhattan_dist_points(near_factory_tile_from_dig_spot, target_dug_tile) == 1

    is_adj, dir_to_fac = adjacent_to_factory(unit_pos, assigned_factory.pos)
    if is_adj:  # if nearby factory + carrying resources: transfer them (reassignment handling) and stop (power after)
        cur_cargo = [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]
        for cargo_amt, r_code in cur_cargo:
            if cargo_amt and is_registry_free(init_turn+actions_counter+1, unit_pos, position_registry, unit.unit_id):
                actions.extend([unit.transfer(dir_to_fac, r_code, cargo_amt)])
                actions_counter += 1
                position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})
                return actions

    # move nearby dedicated spot to be able to transfer stuff, then wait for instructions
    if not np.array_equal(unit_pos, near_factory_tile_from_dig_spot):
        itin_d = make_itinerary_advanced(unit, near_factory_tile_from_dig_spot, game_state, position_registry,
                                         starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
        actions.extend(itin_d["actions"])
        position_registry.update(itin_d["position_registry"])
        return actions

    # theoretically from here, the unit will keep digging and transferring forever, thus not moving unless interrupted
    # a bit dodgy as could be reassigned, but in such case the registry should be updated ... hopefully...
    # for turn in range(init_turn + actions_counter + 1, 1000 + 1):
    for turn in range(init_turn + actions_counter + 1, init_turn + actions_counter + 100):
        # is called every 2 turns, booking for 100 turns is ok
        position_registry[(turn, tuple(unit_pos))] = unit.unit_id

    # compute how much power should be given  (assumes day 60% of the time on the long run)
    sucked_power = int(np.ceil((assist_unit_cfg.DIG_COST * n_desired_digs -
                                0.6 * assist_unit_cfg.CHARGE * (n_desired_digs + 1)) / (n_desired_digs + 1))) * 2

    if assisted_unit.power <= assist_unit_cfg.BATTERY_CAPACITY / 15:  # if unit is low in battery, try to give extra
        sucked_power += int(assist_unit_cfg.BATTERY_CAPACITY / 15)

    if factories_power[assigned_factory.unit_id] > 1000 and assisted_unit.power < 0.8 * assist_unit_cfg.BATTERY_CAPACITY:
        sucked_power += int(assist_unit_cfg.BATTERY_CAPACITY / 30)  # can take much more, factory has plenty

    sucked_power = min((sucked_power, factories_power[assigned_factory.unit_id],
                        unit_cfg.BATTERY_CAPACITY - unit_power,
                        assist_unit_cfg.BATTERY_CAPACITY - assisted_unit.power))

    # need to make a difference between sucked power and transmitted power, as assistant might be full
    transmitted_power = max(0, min(assist_unit_cfg.BATTERY_CAPACITY - assisted_unit.power,
                                   min(unit_power + sucked_power - 10, int(0.9 * (unit_power + sucked_power)))))

    # take corresponding power
    if sucked_power:
        actions.extend([unit.pickup(4, sucked_power, repeat=0)])
        factories_power[assigned_factory.unit_id] -= sucked_power
    else:
        actions.extend([unit.move(0)])  # just to keep everything similarly looking as if we pick, for simplicity
    actions_counter += 1
    # position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    # find future position of assisted_unit
    # assisted_pos_lookup = {(t, (x, y)): u_id for (t, (x, y)), u_id in position_registry.items()
    #                        if u_id == assisted_unit.unit_id and t == init_turn + actions_counter + 1}
    # future_pos_assisted_unit = np.array(list(assisted_pos_lookup.keys())[0][1]) if len(assisted_pos_lookup) else None
    future_pos_assisted_unit = assisted_unit.pos if np.array_equal(assisted_unit.pos, target_dug_tile) else None
    # todo: debug alternative, i.e. actual future position (wonder if should be + 2, power transmitted AFTER move)

    # transfer power
    if transmitted_power and future_pos_assisted_unit is not None:
        resource_type = 4  # power
        transfer_direction = direction_to(near_factory_tile_from_dig_spot, future_pos_assisted_unit)
        actions.extend([unit.transfer(transfer_direction, resource_type, transmitted_power, repeat=0)])
    else:
        actions.extend([unit.move(0)])  # just to keep everything similarly looking as if we transmit, for simplicity

    # actions_counter += 1
    # position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    return actions


# to be used on ice or ore
def go_dig_resource(unit, game_state, position_registry, target_dug_tile, assigned_factory, factories_power, n_min_digs=5,
                    n_desired_digs=8):
    """
    - [move if in the middle of the factory and stop actions there]
    - [pick unit_power]
    - go to tile
    - dig
    - come back nearby factory
    - transfer

    # important to start with unit_power to prevent over picking unit_power which makes game crash)
    :return: [actions]
    """

    # TODO: RECOVERY MANAGEMENT: IF HAVE SOME RESOURCES AND NEARBY FACTORY, DROP THEM!!
    #       ACTUALLY ALL THE ROLES SHOULD START LIKE THAT (except adj-tasks obviously)

    # todo: MASSIVE ISSUE: all the position_registry.update that are NOT following a make_itinerary are actually
    #       not checking for availability of the tile, they just take. They should be able to compromise if
    #       another robot intends to use the tile...

    actions, actions_counter = list(), 0

    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit)

    nearest_factory_tile_from_dug_tile = nearest_factory_tile(target_dug_tile, assigned_factory.pos)
    if np.array_equal(unit_pos, assigned_factory.pos):  # if it is on the central factory tile, move (unsafe)
        itin_d = make_itinerary_advanced(unit, nearest_factory_tile_from_dug_tile, game_state, position_registry,
                                         starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
        if np.array_equal(itin_d["unit_pos"], nearest_factory_tile_from_dug_tile):
            actions.extend(itin_d["actions"])
            position_registry.update(itin_d["position_registry"])
            return actions

    is_ice_dig = game_state.board.ice[target_dug_tile[0], target_dug_tile[1]]
    is_ore_dig = game_state.board.ore[target_dug_tile[0], target_dug_tile[1]]
    assert is_ice_dig or is_ore_dig

    # if nearby factory AND carrying resources transfer them (reassignment handling) and stop (will take unit_power after)
    is_adj, dir_to_fac = adjacent_to_factory(unit_pos, assigned_factory.pos)
    for cargo_amt, r_code in [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]:
        if is_adj and cargo_amt:
            actions.extend([unit.transfer(dir_to_fac, r_code, cargo_amt)])
            actions_counter += 1
            position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})
            return actions

    dist = manhattan_dist_points(unit_pos, target_dug_tile)
    nb_rubble = 60  # proxy rubble for simplicity, should check actual rubble on map
    power_cost_move = int(2 * dist * (unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * nb_rubble))
    desired_buffer = int(unit_cfg.BATTERY_CAPACITY / 30)

    # # itinerary computation made before, so we know how much it costs to get there
    # TODO: care: it's assumed to be the same cost to come back to the factory (x2 factor ), which might be VERY wrong\
    # TODO: could use proxy to compute the cost... can't use itinerary, because we might take unit_power, which
    #       f*cks up the position registry because we're then late by one turn
    # itin_d = make_itinerary_advanced(unit, target_dug_tile, game_state, position_registry,
    #                                  starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
    # unit_pos_after_moving, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
    # power_cost_move = 2 * power_cost
    # desired_buffer = int(unit_cfg.BATTERY_CAPACITY / 30)

    # pick power if necessary
    res = take_power(unit, game_state, position_registry, assigned_factory, factories_power, power_cost_move, unit_pos,
                     unit_power, unit_cargo, starting_turn=init_turn+actions_counter, n_min_digs=n_min_digs,
                     n_desired_digs=n_desired_digs, desired_buffer=desired_buffer, slow_if_low_power=False)

    if res["stop_after"]:
        return res["actions"]

    # apply res to current context
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit, res["unit_pos"], res["unit_power"], res["unit_cargo"])
    actions.extend(res["actions"])
    position_registry.update(res["position_registry"])
    actions_counter += res["actions_counter"]

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    # actually go to target
    itin_d = make_itinerary_advanced(unit, target_dug_tile, game_state, position_registry,
                                     starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
    unit_pos, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
    actions.extend(itin_d["actions"])
    position_registry.update(itin_d["position_registry"])
    actions_counter += itin_d["actions_counter"]
    unit_power -= power_cost
    if not np.array_equal(unit_pos, target_dug_tile):
        return actions  # could not find a way... abort other steps for now

    # dig
    # TODO: improve power computation
    n_dig_eventually = int(np.floor(0.9 * (unit_power - power_cost_move/2) / unit_cfg.DIG_COST))
    actions.extend([unit.dig(repeat=0, n=n_dig_eventually)])
    unit_power -= unit_cfg.DIG_COST * n_dig_eventually
    for i in range(n_dig_eventually):
        actions_counter += 1
        position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    unit_power += count_day_turns(
        turn_start=init_turn, turn_end=init_turn + actions_counter) * unit_cfg.CHARGE + unit_cfg.ACTION_QUEUE_POWER_COST
    anticipated_cargo = [(unit.cargo.ice + is_ice_dig * n_dig_eventually * unit_cfg.DIG_RESOURCE_GAIN, 0),
                         (unit.cargo.ore + is_ore_dig * n_dig_eventually * unit_cfg.DIG_RESOURCE_GAIN, 1),
                         (unit.cargo.water, 2), (unit.cargo.metal, 3)]
    actions.extend(go_to_factory(unit, game_state, position_registry, assigned_factory, unit_pos,
                                 starting_turn=init_turn + actions_counter, unit_power=unit_power,
                                 unit_cargo=anticipated_cargo))

    return actions


def go_dig_rubble(unit, game_state, position_registry, assigned_factory, factories_power, rubble_tiles_being_dug,
                  tiles_scores=None, n_min_digs=5, n_desired_digs=8):

    obs_n_tiles = 10
    if tiles_scores is None:
        tiles_scores = score_rubble_tiles_to_dig(game_state, assigned_factory, obs_n=obs_n_tiles)

    actions, actions_counter = list(), 0
    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit)

    # change score metric to slightly favor tiles to dug nearby unit_pos
    tiles_scores = score_rubble_add_proximity_penalty_to_tiles_to_dig(tiles_scores, unit_pos)
    tiles_scores = {t: sc for t, sc in tiles_scores.items() if tuple(t) not in rubble_tiles_being_dug.values()}
    if not len(tiles_scores):
        tiles_scores = score_rubble_tiles_to_dig(game_state, assigned_factory, obs_n=14)
        tiles_scores = score_rubble_add_proximity_penalty_to_tiles_to_dig(tiles_scores, unit_pos)
        tiles_scores = {t: sc for t, sc in tiles_scores.items() if tuple(t) not in rubble_tiles_being_dug.values()}
        if not len(tiles_scores):
            actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=3,
                                          unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
            # actions.extend([unit.recharge(unit_cfg.BATTERY_CAPACITY)])
            # all rubble has been dug... should do something else...
            return actions

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    ordered_tiles = np.array(sorted(tiles_scores.keys(), key=lambda t: tiles_scores[(t[0], t[1])]))
    best_candidate_target_dug_tile = ordered_tiles[0]  # need one only to move away if in center of factory

    nearest_factory_tile_from_dug_tile = nearest_factory_tile(best_candidate_target_dug_tile, assigned_factory.pos)
    if np.array_equal(unit_pos, assigned_factory.pos):  # if it is on the central factory tile, move (unsafe)
        itin_d = make_itinerary_advanced(unit, nearest_factory_tile_from_dug_tile, game_state,
                                         position_registry, starting_turn=init_turn + actions_counter,
                                         unit_pos=unit_pos)
        if np.array_equal(itin_d["unit_pos"], nearest_factory_tile_from_dug_tile):
            actions.extend(itin_d["actions"])
            position_registry.update(itin_d["position_registry"])
            return actions

    # if nearby factory AND carrying resources transfer them (reassignment handling) and stop (will take power after)
    is_adj, dir_to_fac = adjacent_to_factory(unit_pos, assigned_factory.pos)
    for cargo_amt, r_code in [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]:
        if is_adj and cargo_amt:
            actions.extend([unit.transfer(dir_to_fac, r_code, cargo_amt)])
            actions_counter += 1
            position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})
            return actions

    dist = manhattan_dist_points(unit_pos, best_candidate_target_dug_tile)
    dist2 = manhattan_dist_points(unit_pos, assigned_factory.pos)
    nb_rubble = 60  # proxy rubble for simplicity, should check actual rubble on map
    power_cost_move = int((dist + dist2) * (unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * nb_rubble))
    desired_buffer = int(unit_cfg.BATTERY_CAPACITY / 30)

    # if game_state.real_env_steps >= 99 and unit.unit_id == "unit_38":
    #     pass

    # # itinerary computation made before, so we know how much it costs to get there
    # TODO: care: it's assumed to be the same cost to come back to the factory (x2 factor), which might be VERY wrong
    # itin_d = make_itinerary_advanced(unit, target_dug_tile, game_state, position_registry,
    #                                  starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
    # unit_pos_after_moving, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
    # power_cost_move = 2 * power_cost
    # desired_buffer = int(unit_cfg.BATTERY_CAPACITY / 30)

    # n_desired_digs = min(n_desired_digs, n_necessary_digs)
    # n_min_digs = min(n_desired_digs, n_min_digs)

    # pick power if necessary
    slow_if_low_power = True  # digging is not an early priority, pick low amount if factory is low
    res = take_power(unit, game_state, position_registry, assigned_factory, factories_power, power_cost_move, unit_pos,
                     unit_power, unit_cargo, starting_turn=init_turn + actions_counter, n_min_digs=n_min_digs,
                     n_desired_digs=n_desired_digs, desired_buffer=desired_buffer, slow_if_low_power=slow_if_low_power)

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    if res["stop_after"]:
        return res["actions"]

    # apply res to current context
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit, res["unit_pos"], res["unit_power"], res["unit_cargo"])
    actions.extend(res["actions"])
    position_registry.update(res["position_registry"])
    actions_counter += res["actions_counter"]

    # find actual tile we want to go to
    target_dug_tile = None
    for t in ordered_tiles:
        candidate_target_dug_tile = np.array(t)
        itin_d = make_itinerary_advanced(unit, candidate_target_dug_tile, game_state, position_registry,
                                         starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
        unit_pos_after_moving, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
        if not np.array_equal(unit_pos_after_moving, candidate_target_dug_tile):
            continue  # can't find a way there, move on to another candidate

        # check how long the tile is available after arriving there
        dig_availability = [is_registry_free(init_turn + actions_counter + itin_d["actions_counter"] + i + 1, t,
                                             position_registry, unit.unit_id) for i in range(n_desired_digs)]
        tile_available_counter = 0
        while tile_available_counter < len(dig_availability):
            if not dig_availability[tile_available_counter]:
                break
            tile_available_counter += 1

        if tile_available_counter >= n_min_digs:  # todo: not sure if need +1 here...
            target_dug_tile = candidate_target_dug_tile
            # rubble_tiles_being_dug[unit.unit_id] = tuple(t)
            break  # candidate target_dug tile becomes our dig objective
        else:
            continue  # candidate_target_dug_tile is not available long enough, we move on

    if target_dug_tile is None:  # could not find a tile to dig. Wait 4 turns and hope it gets better?
        n_turns_static = 4
        actions.extend([unit.move(direction=0, repeat=0, n=n_turns_static)])
        for i in range(n_turns_static):
            position_registry[(init_turn + i + 1, tuple(unit_pos))] = unit.unit_id
        return actions

    is_ice_dig = game_state.board.ice[target_dug_tile[0], target_dug_tile[1]]
    is_ore_dig = game_state.board.ore[target_dug_tile[0], target_dug_tile[1]]
    assert (not is_ice_dig) and (not is_ore_dig)

    # go to target_dug_tile... have to recompute itinerary because we might have taken power
    # itin_d = make_itinerary_advanced(unit, target_dug_tile, game_state, position_registry,
    #                                  starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
    actions.extend(itin_d["actions"])
    unit_pos = target_dug_tile
    unit_power -= itin_d["power_cost"]
    rubble_tiles_being_dug[unit.unit_id] = tuple(target_dug_tile)
    position_registry.update(itin_d["position_registry"])
    actions_counter += itin_d["actions_counter"]
    # if not np.array_equal(unit_pos, target_dug_tile):
    #     return actions  # could not find a way... abort other steps for now

    # dig a reasonable amount of times (function of power and quantity of rubble)
    # TODO: improve power computation
    n_necessary_digs = int(np.ceil(
        game_state.board.rubble[target_dug_tile[0], target_dug_tile[1]] / unit_cfg.DIG_RUBBLE_REMOVED))
    n_dig_eventually = max(n_min_digs - 2, max(1, min(tile_available_counter, min(int(np.floor(
        (unit_power - power_cost_move/2) / unit_cfg.DIG_COST)), n_necessary_digs))))
    if n_dig_eventually < 1:
        pass
    # n_dig_eventually = n_necessary_digs if abs(n_necessary_digs - n_dig_eventually) <= 1 else n_dig_eventually
    actions.extend([unit.dig(repeat=0, n=n_dig_eventually)])
    for i in range(n_dig_eventually):
        actions_counter += 1
        position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    # go back to factory? will go back anyway if not enough power to keep going...
    # unit_pos = nearest_factory_tile_from_dug_tile

    return actions


def go_resist(unit, game_state, position_registry, op_unit, assigned_factory, threat_desc, allow_weaker=False):
    # todo: code several turns in a row, and update only if required (power is important when fighting!)
    # todo: make sure we don't move into a factory (even allied, we want to fight!)
    if not allow_weaker:
        assert is_unit_stronger(unit, op_unit)
    actions, actions_counter = list(), 0
    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    unit_pos, power = unit.pos, unit.power
    dist_to_threat = threat_desc[1]

    pos_diff = op_unit.pos - unit.pos  # can be [(+-1, 0), (0, +-1) or (+-1, +-1) or (+-2, 0) or (0, +-2)]
    if dist_to_threat == 1 and \
            is_registry_free(init_turn + actions_counter, op_unit.pos, position_registry, unit.unit_id) and \
            game_state.board.factory_occupancy_map[op_unit.pos[0], op_unit.pos[1]] == -1:
        dir_code = get_direction_code_from_delta(tuple(pos_diff))   # simply go to it!
        position_registry[(init_turn + actions_counter + 1, tuple(unit_pos))] = unit.unit_id
        actions.extend([unit.move(dir_code)])
        return actions

    dx, dy = tuple(pos_diff)
    dx, dy = int(dx/abs(dx)) if dx else 0, int(dy/abs(dy)) if dy else 0
    candidates_deltas = list()
    if abs(dx):
        candidates_deltas.append((dx, 0))
    if abs(dy):
        candidates_deltas.append((0, dy))
    candidates_deltas = sorted(candidates_deltas, key=lambda delta: game_state.board.rubble[delta[0], delta[1]])

    for delta in candidates_deltas:
        new_pos = unit_pos + np.array(delta)
        if is_registry_free(init_turn + actions_counter, new_pos, position_registry, unit.unit_id) and \
                game_state.board.factory_occupancy_map[new_pos[0], new_pos[1]] == -1:
            dir_code = get_direction_code_from_delta(delta)
            position_registry[(init_turn + actions_counter + 1, tuple(new_pos))] = unit.unit_id
            actions.extend([unit.move(dir_code)])
            return actions

    # if we could not find something that does not kill anyone... well we wander around ?
    # todo: prioritise moving in this context (we usually prioritise NOT moving)
    actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=1, unit_pos=unit_pos,
                                  starting_turn=init_turn + actions_counter, prioritise_static=False))
    return actions


def go_resist2(unit, game_state, position_registry, op_unit, assigned_factory, threat_dist, next_unit_action,
               factories_power, unit_pos=None, unit_power=None, unit_cargo=None, is_aggressive=False):

    actions, actions_counter = list(), 0
    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit, unit_pos, unit_power, unit_cargo)
    player_me = "player_0" if unit.unit_id in game_state.units["player_0"].keys() else "player_1"
    player_op = "player_1" if player_me == "player_0" else "player_0"
    opponent_factory_tiles = np.array([f.pos for f in game_state.factories[player_op].values()])
    threat_level = threat_category(unit, op_unit)
    all_deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
    random.shuffle(all_deltas)

    # pick power if it helps the resistance
    cur_cheb_dist_to_factory = chebyshev_dist_points(assigned_factory.pos, unit_pos)
    if cur_cheb_dist_to_factory <= 1 and threat_level == "threaten_losing":
        needed_power_to_dominate = op_unit.power + unit_cfg.ACTION_QUEUE_POWER_COST - unit_power + 1
        if 0 < needed_power_to_dominate < factories_power[assigned_factory.unit_id] and \
                is_registry_free(init_turn + actions_counter + 1, unit_pos, position_registry, unit.unit_id):
            actions_counter += 1
            position_registry[(init_turn + actions_counter, tuple(unit_pos))] = unit.unit_id
            actions.extend([unit.pickup(4, needed_power_to_dominate)])

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    if threat_level == "winning":
        return None

    # todo: ensure the unit is added as a fighter (to be cleared once not threaten anymore)
    elif threat_level == "threaten_winning":
        if threat_dist >= 2:
            return None
        op_direction = direction_to(unit.pos, op_unit.pos)
        if next_unit_action is None or not (next_unit_action[0] == 0 and next_unit_action[1] == op_direction):
            distances_to_factories = manh_dist_to_factory_vect_point(opponent_factory_tiles, op_unit.pos)
            if is_registry_free(init_turn + actions_counter + 1, op_unit.pos, position_registry, unit.unit_id) and (
                    not len(distances_to_factories) or np.min(distances_to_factories) > 0):
                delete_value_from_dict(position_registry, unit.unit_id)
                position_registry[(init_turn + actions_counter + 1, tuple(op_unit.pos))] = unit.unit_id
                actions.extend([unit.move(op_direction)])
                return actions
            else:
                delete_value_from_dict(position_registry, unit.unit_id)
                actions.extend(wander_n_turns(
                    unit, game_state, position_registry, assigned_factory, n_wander=1, unit_pos=unit_pos,
                    starting_turn=init_turn + actions_counter, prioritise_static=False))
                return actions
        return None

    elif threat_level == "threaten_losing":
        if threat_dist == 2:
            dx, dy = op_unit.pos - unit_pos
            op_directions = []
            if abs(dx):
                op_directions.append(get_direction_code_from_delta((int(dx/abs(dx)), 0)))
            if abs(dy):
                op_directions.append(get_direction_code_from_delta((0, int(dy/abs(dy)))))
            if next_unit_action is not None and (next_unit_action[0] == 0 and next_unit_action[1] in op_directions):
                # would rather go somewhere else... i.e. any direction except the ones with a dist one with the opponent
                options = sorted(all_deltas, key=lambda d: 1 if manhattan_dist_points(
                    unit_pos + np.array(d), op_unit.pos) != 1 else 100)
            else:
                return None
        elif threat_dist > 2:
            return None
        elif threat_dist == 1:
            if chebyshev_dist_points(unit_pos, assigned_factory.pos) <= 1:
                return None
            else:
                if next_unit_action is None or next_unit_action[0] != 0 or next_unit_action[1] == 0:
                    # would rather go somewhere else, priority given to going towards factory
                    options = sorted([d for d in all_deltas if d != (0, 0)],
                                     key=lambda d: manh_dist_to_factory_points(unit_pos + np.array(d),
                                                                               assigned_factory.pos)) + [(0, 0)]
                else:
                    return None
        else:
            raise NotImplementedError("how?")

    elif threat_level == "losing":
        if threat_dist == 2:
            dx, dy = op_unit.pos - unit_pos
            op_directions = []
            if abs(dx):
                op_directions.append(get_direction_code_from_delta((int(dx/abs(dx)), 0)))
            if abs(dy):
                op_directions.append(get_direction_code_from_delta((0, int(dy/abs(dy)))))
            if next_unit_action is not None and (next_unit_action[0] == 0 and next_unit_action[1] in op_directions):
                # would rather go somewhere else... i.e. any direction except the ones with a dist one with the opponent
                options = sorted(all_deltas, key=lambda d: 1 if manhattan_dist_points(
                    unit_pos + np.array(d), op_unit.pos) != 1 else 100)
            else:
                return None
        elif threat_dist > 2:
            return None
        elif threat_dist == 1:
            op_direction = direction_to(unit.pos, op_unit.pos)
            if chebyshev_dist_points(unit_pos, assigned_factory.pos) <= 1:
                if next_unit_action is not None and (next_unit_action[0] == 0 and next_unit_action[1] == op_direction):
                    # would rather go somewhere else than jumping to death
                    options = sorted(
                        all_deltas, key=lambda d: 10 if np.array_equal(unit_pos + np.array(d), op_unit.pos) else 1)
                else:
                    return None
            else:
                if next_unit_action is None or next_unit_action[0] != 0 or next_unit_action[1] == 0 or \
                        (next_unit_action[0] == 0 and next_unit_action[1] == op_direction):
                    # would rather go somewhere else, priority given to going towards factory (no stay, not toward him)
                    options = sorted([d for d in all_deltas if d != (0, 0)], key=lambda d: manh_dist_to_factory_points(
                        unit_pos + np.array(d), assigned_factory.pos) if not np.array_equal(
                        unit_pos + np.array(d), op_unit.pos) else 1000) + [(0, 0)]
                else:
                    return None
        else:
            raise NotImplementedError("how?")

    else:
        raise NotImplementedError(f"unknown threat level: {threat_level}")

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    for delta in options:
        new_pos = unit_pos + np.array(delta)

        if (not 0 <= new_pos[0] < 48) or (not 0 <= new_pos[1] < 48):
            continue  # next move not in the map

        distances_to_factories = manh_dist_to_factory_vect_point(opponent_factory_tiles, new_pos)
        if len(distances_to_factories) and np.min(distances_to_factories) == 0:
            continue  # don't go through an opponent factory

        if not is_registry_free(init_turn + actions_counter + 1, new_pos, position_registry, unit.unit_id):
            continue

        dir_code = get_direction_code_from_delta(delta)
        delete_value_from_dict(position_registry, unit.unit_id)
        position_registry[(init_turn + actions_counter + 1, tuple(new_pos))] = unit.unit_id
        actions.extend([unit.move(dir_code)])
        return actions

    delete_value_from_dict(position_registry, unit.unit_id)
    # if we could not find something that does not kill anyone... well we wander around ?
    actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=1, unit_pos=unit_pos,
                                  starting_turn=init_turn + actions_counter, prioritise_static=False))
    return actions


def go_bully(unit, game_state, position_registry, assigned_factory, bullies_register, factories_power, my_rest_tiles,
             op_rest_tiles, resources_bully_tiles, lichen_bully_tiles, unit_pos=None, unit_power=None, unit_cargo=None):

    actions, actions_counter = list(), 0
    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit, unit_pos, unit_power, unit_cargo)

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    # if nearby factory AND carrying resources transfer them (reassignment handling) and stop (will take power after?)
    is_adj, dir_to_fac = adjacent_to_factory(unit_pos, assigned_factory.pos)
    for cargo_amt, r_code in [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]:
        if is_adj and cargo_amt:
            actions.extend([unit.transfer(dir_to_fac, r_code, cargo_amt)])
            actions_counter += 1
            position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})
            return actions

    if factories_power[assigned_factory.unit_id] > 800 and chebyshev_dist_points(assigned_factory.pos, unit_pos) <= 1:
        if is_registry_free(init_turn + actions_counter + 1, unit_pos, position_registry, unit.unit_id):
            sucked_power = min(unit_cfg.BATTERY_CAPACITY - unit_power, int(0.5 * factories_power[assigned_factory.unit_id]))
            actions.extend([unit.pickup(4, sucked_power)])
            actions_counter += 1

    resources_bully_tiles = sorted(resources_bully_tiles,
                                   key=lambda t: manhattan_dist_points(unit_pos, np.array(t)))
    lichen_bully_tiles = sorted(lichen_bully_tiles,
                                key=lambda t: manhattan_dist_points(unit_pos, np.array(t)))
    op_rest_tiles = sorted(op_rest_tiles, key=lambda t: manhattan_dist_points(unit_pos, np.array(t)))
    my_rest_tiles = sorted(my_rest_tiles, key=lambda t: manhattan_dist_points(unit_pos, np.array(t)))

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
    #     pass

    # need to ensure bully is registered, or to wank somewhere else
    try:
        cur_bullied_tile = bullies_register[unit.unit_id]
    except KeyError:
        cur_bullied_tile = None

    if cur_bullied_tile is None:
        # go to the nearest unregistered rest tile (op factory)
        for tile in op_rest_tiles:
            if tile not in bullies_register.values():
                cur_bullied_tile = tile
                bullies_register[unit.unit_id] = cur_bullied_tile
                break

    if cur_bullied_tile is None:  # if still none, go to the nearest unregistered rest tile (my factories)
        for tile in my_rest_tiles:
            if tile not in bullies_register.values():
                cur_bullied_tile = tile
                bullies_register[unit.unit_id] = cur_bullied_tile
                break

    if cur_bullied_tile is None:  # nowhere to bully...
        actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=3,
                                      unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
        return actions

    bully_situation = "unknown"
    if bullies_register[unit.unit_id] in my_rest_tiles:
        bully_situation = "home_rest"
    if bullies_register[unit.unit_id] in op_rest_tiles:
        bully_situation = "away_rest"
    if bullies_register[unit.unit_id] in resources_bully_tiles or bullies_register[unit.unit_id] in lichen_bully_tiles:
        bully_situation = "active_bullying"

    # flex proxy to assess if the unit is able to fight!
    proxy_end_game = max(game_state.real_env_steps - 900, 0) / 100

    inactive_pow_threshold = 0.2 * (1. - proxy_end_game) * unit_cfg.BATTERY_CAPACITY
    active_pow_threshold = 0.6 * (1. - proxy_end_game) * unit_cfg.BATTERY_CAPACITY
    if manhattan_dist_points(np.array(bullies_register[unit.unit_id]), unit_pos) <= 5 and \
            unit_power >= active_pow_threshold:
        if bully_situation == "home_rest":
            for tile in op_rest_tiles:
                if tile not in bullies_register.values():
                    cur_bullied_tile = tile
                    bullies_register[unit.unit_id] = cur_bullied_tile
                    break
        if bully_situation == "away_rest":
            for tile in resources_bully_tiles + lichen_bully_tiles:
                if tile not in bullies_register.values() and manhattan_dist_points(np.array(tile), unit_pos) <= 16:
                    # only go to active bullying if it's not super far
                    cur_bullied_tile = tile
                    bullies_register[unit.unit_id] = cur_bullied_tile
                    break
    if bully_situation == "active_bullying" and unit_power <= inactive_pow_threshold:
        for tile in op_rest_tiles + my_rest_tiles:
            if tile not in bullies_register.values():
                cur_bullied_tile = tile
                bullies_register[unit.unit_id] = cur_bullied_tile
                break

    # weird flex proxy to assess if the unit is able to fight!
    should_fight = unit_power >= (1000 - game_state.real_env_steps) * (unit_cfg.DIG_COST + unit_cfg.MOVE_COST) / 2.3

    if should_fight:
        lichen_attack_actions = attack_opponent_lichen(
            unit, game_state, position_registry, assigned_factory, control_area=3,
            starting_turn=init_turn+actions_counter, unit_pos=unit_pos, unit_power=unit_power, unit_cargo=unit_cargo)

        if lichen_attack_actions is not None:
            actions.extend(lichen_attack_actions)
            return actions
        # scrutinised_tiles = prioritise_attack_tiles_nearby(
        #     unit, game_state, position_registry, unit_pos=unit_pos, starting_turn=init_turn+actions_counter,
        #     control_area=(3 if bully_situation == "away_rest" else 2))
        # for attacked_tile in scrutinised_tiles:
        #     itin_d = make_itinerary_advanced(unit, attacked_tile, game_state, position_registry,
        #                                      starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
        #     if np.array_equal(attacked_tile, itin_d["unit_pos"]) and is_registry_free(
        #             init_turn + actions_counter + itin_d["actions_counter"], attacked_tile, position_registry,
        #             unit.unit_id):
        #         unit_pos, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
        #         if itin_d["power_cost"] > unit_power:
        #             actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=3,
        #                                           unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
        #             return actions
        #         actions.extend(itin_d["actions"])
        #         position_registry.update(itin_d["position_registry"])
        #         actions_counter += itin_d["actions_counter"]
        #
        #         # then dig
        #         actions_counter += 1
        #         position_registry[(init_turn + actions_counter, tuple(unit_pos))] = unit.unit_id
        #         actions.extend([unit.dig(repeat=0, n=1)])
        #         return actions

    # unit has already been registered, if no extra fighting plan just stick around
    guarded_tile = np.array(bullies_register[unit.unit_id])
    if not np.array_equal(guarded_tile, unit_pos):
        itin_d = make_itinerary_advanced(unit, guarded_tile, game_state, position_registry,
                                         starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
        if itin_d["power_cost"] > unit_power:
            actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=3,
                                          unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
            return actions
        unit_pos, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
        actions.extend(itin_d["actions"])
        position_registry.update(itin_d["position_registry"])
        actions_counter += itin_d["actions_counter"]
        return actions

    # then wait there and charge
    actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=3,
                                  unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
    return actions


def attack_opponent_lichen(unit, game_state, position_registry, assigned_factory, control_area=3,
                           starting_turn=None, unit_pos=None, unit_power=None, unit_cargo=None):

    actions, actions_counter, pos_registry_new = list(), 0, dict()
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit, unit_pos, unit_power, unit_cargo)
    unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
    init_turn = game_state.real_env_steps if starting_turn is None else starting_turn

    scrutinised_tiles = prioritise_attack_tiles_nearby(
        unit, game_state, position_registry, unit_pos=unit_pos, starting_turn=init_turn + actions_counter,
        control_area=control_area)
    for attacked_tile in scrutinised_tiles:
        itin_d = make_itinerary_advanced(unit, attacked_tile, game_state, position_registry,
                                         starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
        if np.array_equal(attacked_tile, itin_d["unit_pos"]) and is_registry_free(
                init_turn + actions_counter + itin_d["actions_counter"], attacked_tile, position_registry,
                unit.unit_id):
            unit_pos, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
            if itin_d["power_cost"] > unit_power:
                actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=3,
                                              unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
                return actions
            actions.extend(itin_d["actions"])
            position_registry.update(itin_d["position_registry"])
            actions_counter += itin_d["actions_counter"]

            # then dig
            actions_counter += 1
            position_registry[(init_turn + actions_counter, tuple(unit_pos))] = unit.unit_id
            actions.extend([unit.dig(repeat=0, n=1)])
            return actions
    return None

# def empty_cargo(unit, game_state, position_registry, assigned_factory, unit_pos=None, starting_turn=None,
#                 anticipated_cargo=None):
#     """
#     Return the actions that empty the cargo.
#     Care, at this stage does not return any extra info about turn counter / power etc
#     Care, at this stage simply won't do it if position_registry is not free to do so...
#
#     :return: [empty_cargo_actions]
#     """
#     if anticipated_cargo is None:
#         anticipated_cargo = [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]
#
#     actions, actions_counter = list(), 0
#     unit_pos = unit.pos if unit_pos is None else unit_pos
#     init_turn = game_state.real_env_steps if starting_turn is None else starting_turn
#
#     # if nearby factory AND carrying resources transfer them (reassignment handling) and stop (will take power after)
#     is_adj, dir_to_fac = adjacent_to_factory(unit_pos, assigned_factory.pos)
#     for cargo_amt, r_code in anticipated_cargo:
#         if is_adj and cargo_amt and is_registry_free(init_turn + actions_counter + 1, unit_pos, position_registry,
#                                                      unit.unit_id):
#             actions.extend([unit.transfer(dir_to_fac, r_code, cargo_amt)])
#             actions_counter += 1
#             position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})
#
#     return actions


def take_power(unit, game_state, position_registry, assigned_factory, factories_power, power_cost_move, unit_pos=None,
               unit_power=None, unit_cargo=None, starting_turn=None, n_min_digs=5, n_desired_digs=8, desired_buffer=0,
               slow_if_low_power=False):
    """
    Ensure power is taken IF required, in proportions that are reasonable.
    If in need for power (as expressed by n_min_digs and n_desired_digs) and including expected power_cost_move:
        If needed go to factory tile (then stop instructions, need to be called again later on)
        Take appropriate power as assessed by some heuristics (then don't stop instructions)

    :return:  {"actions": actions, "position_registry": pos_registry_new,
                "unit_pos": unit_pos, "unit_power": unit_power, "unit_cargo": unit_cargo,
                "actions_counter": actions_counter, "stop_after": True/False}
    """
    actions, actions_counter, pos_registry_new = list(), 0, dict()
    unit_pos, unit_power, unit_cargo = get_pos_power_cargo(unit, unit_pos, unit_power, unit_cargo)
    unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
    init_turn = game_state.real_env_steps if starting_turn is None else starting_turn

    # pick power?
    min_power = min(unit_cfg.DIG_COST * n_min_digs + power_cost_move, unit_cfg.BATTERY_CAPACITY)
    desired_power = min(unit_cfg.DIG_COST * n_desired_digs + power_cost_move+desired_buffer, unit_cfg.BATTERY_CAPACITY)
    picked_power = 0
    cur_cheb_dist_to_factory = chebyshev_dist_points(assigned_factory.pos, unit_pos)
    if unit_power < desired_power:
        if factories_power[assigned_factory.unit_id] >= desired_power - unit_power:
            picked_power = desired_power - unit_power
        elif unit_power > min_power:
            picked_power = 0  # let's not pick if we have a reasonable amount
        elif factories_power[assigned_factory.unit_id] >= min_power - unit_power:
            picked_power = (min_power - unit_power) if slow_if_low_power else factories_power[assigned_factory.unit_id]
        elif cur_cheb_dist_to_factory <= 1:
            # care: will directly modify the registry...
            actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=3,
                                          unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
            # return actions
            return {"actions": actions, "position_registry": pos_registry_new,
                    "unit_pos": unit_pos, "unit_power": unit_power, "unit_cargo": unit_cargo,
                    "actions_counter": actions_counter, "stop_after": True}

        if cur_cheb_dist_to_factory > 1 and (unit_power < min_power or picked_power):
            # care: will directly modify the registry...
            actions.extend(go_to_factory(
                unit, game_state, position_registry, assigned_factory, unit_pos=unit_pos,
                starting_turn=init_turn + actions_counter))
            # return actions
            return {"actions": actions, "position_registry": pos_registry_new,
                    "unit_pos": unit_pos, "unit_power": unit_power, "unit_cargo": unit_cargo,
                    "actions_counter": actions_counter, "stop_after": True}

        if picked_power:
            if factories_power[assigned_factory.unit_id] > 900:  # can take much more, factory has plenty
                picked_power = min(unit_cfg.BATTERY_CAPACITY - unit_power,
                                   max(picked_power, int(0.10 * (factories_power[assigned_factory.unit_id] - 800))))

            if is_registry_free(init_turn + actions_counter + 1, unit_pos, position_registry, unit.unit_id):
                actions.extend([unit.pickup(4, picked_power)])
                position_registry.update({(init_turn + actions_counter + 1, tuple(unit_pos)): unit.unit_id})
                actions_counter += 1
                factories_power[assigned_factory.unit_id] -= picked_power
                unit_power += picked_power
            else:
                # care: will directly modify the registry...
                # actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory,
                #                               n_wander=2, unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
                actions.extend(go_to_factory(unit, game_state, position_registry, assigned_factory, unit_pos=unit_pos,
                                             starting_turn=init_turn + actions_counter))
                return {"actions": actions, "position_registry": pos_registry_new,
                        "unit_pos": unit_pos, "unit_power": unit_power, "unit_cargo": unit_cargo,
                        "actions_counter": actions_counter, "stop_after": True}

    return {"actions": actions, "position_registry": pos_registry_new,
            "unit_pos": unit_pos, "unit_power": unit_power, "unit_cargo": unit_cargo,
            "actions_counter": actions_counter, "stop_after": False}


def go_to_factory(unit, game_state, position_registry, assigned_factory, unit_pos=None, starting_turn=None,
                  stay_extra_turn=True, unit_power=None, unit_cargo=None,
                  can_reassign=False):
    """
    - go back to the nearest tile within factory
    - [drop stuff being carried]

    (that's it... Idea is to call that one if running out of power when digging for example...)

    :return: [actions]
    """

    if unit_cargo is None:
        unit_cargo = [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]

    if unit_power is None:
        unit_power = unit.power

    actions, actions_counter = list(), 0
    unit_pos = unit.pos if unit_pos is None else unit_pos

    unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
    init_turn = game_state.real_env_steps if starting_turn is None else starting_turn

    cargo_counter = 0
    for cargo_amt, r_code in unit_cargo:
        cargo_counter += 1 if cargo_amt else 0
        # todo: if adjacent to factory, transfer cargo to factory already

    # return {"actions": actions, "power_cost": power_cost, "position_registry": pos_registry_new,
    #         "unit_pos": unit_pos, "actions_counter": len(path)}

    excluded_tiles = list()
    while len(excluded_tiles) < 8:
        near_factory_tile = nearest_factory_tile(unit_pos, assigned_factory.pos, excluded_tiles=excluded_tiles)

        manh_dist_to_tile = manhattan_dist_points(near_factory_tile, unit_pos)
        tile_stay_length = cargo_counter + (1 if stay_extra_turn else 0) + 1
        assumed_availability = [is_registry_free(init_turn + manh_dist_to_tile + i, near_factory_tile,
                                                 position_registry, unit.unit_id) for i in range(tile_stay_length + 2)]
        consecutive_availability = [all(assumed_availability[i:(tile_stay_length +i)]) for i in range(2+1)]
        if sum(consecutive_availability) == 0:
            excluded_tiles.append(tuple(near_factory_tile))
            continue  # skip path making, too slow unlikely to succeed...

        itin_d = make_itinerary_advanced(unit, near_factory_tile, game_state, position_registry,
                                         starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
        candidate_unit_pos, power_cost = itin_d["unit_pos"], itin_d["power_cost"]

        # if game_state.real_env_steps in monitored_turns and unit.unit_id == "unit_50":
        #     end_time = time.time()
        #     local_time = str(end_time - start_time)
        #     pass
        #     start_time = time.time()

        if power_cost + unit_cfg.ACTION_QUEUE_POWER_COST >= unit_power:  # unless not enough power (recharge if so)
            # # todo: enquire if enough space to stay around before attempting to charge... Will block way for some robots
            # actions.extend([unit.recharge(power_cost + unit_cfg.ACTION_QUEUE_POWER_COST)])
            actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=3,
                                          unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
            return actions
            # excluded_tiles.append(tuple(near_factory_tile))
            # continue

        if np.array_equal(candidate_unit_pos, near_factory_tile) and all(
                [is_registry_free(init_turn + actions_counter + itin_d["actions_counter"] + i + 1,
                                  near_factory_tile, position_registry, unit.unit_id) for i in
                 range(cargo_counter + (1 if stay_extra_turn else 0))]):
            actions.extend(itin_d["actions"])
            position_registry.update(itin_d["position_registry"])
            actions_counter += itin_d["actions_counter"]
            unit_pos = candidate_unit_pos
            break
            # return actions

        # if it didn't work, try to make it work with next tile...
        excluded_tiles.append(tuple(near_factory_tile))

    if len(actions) == 0:  # we failed to find a way to unload cargo and to sit around for one turn
        actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=3,
                                      unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
        return actions

    # if unit.unit_id == "unit_50" and game_state.real_env_steps == 866:
    #     end_time = time.time()
    #     local_time = str(end_time - start_time)
    #     pass
    #     start_time = time.time()

    # transfer carried resources
    for cargo_amt, r_code in unit_cargo:
        if cargo_amt:
            actions.extend([unit.transfer(0, r_code, cargo_amt)])
            actions_counter += 1
            position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    if stay_extra_turn:
        # cheeky extra book to take power... (but power NOT taken, assumed to be first action next)
        position_registry.update({(init_turn + actions_counter + 1, tuple(unit_pos)): unit.unit_id})

    return actions


def wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=3, unit_pos=None,
                   starting_turn=None, prioritise_static=True, replace_static_by_charge=True):
    # function to stick around and trying to survive when nothing seems to work (i.e. wait for n turns)
    # todo: if move of +dx, next favored move should be -dx...
    # TODO: unload cargo ONLY if nearby factory (did i forget this ?)
    unit_pos = unit.pos if unit_pos is None else unit_pos
    unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
    init_turn = game_state.real_env_steps if starting_turn is None else starting_turn
    actions, actions_counter = list(), 0

    cargo_options = [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]
    to_be_cargod = [(cargo_amt, r_code) for cargo_amt, r_code in cargo_options if cargo_amt]

    # a bit silly to guess that here...
    player_me = "player_0" if unit.unit_id in game_state.units["player_0"].keys() else "player_1"
    player_op = "player_1" if player_me == "player_0" else "player_0"
    opponent_factory_tiles = np.array([f.pos for f in game_state.factories[player_op].values()])

    # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
    #     pass

    idx_cargo = 0
    for i in range(n_wander):

        # define explore options and their order
        explore_options = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        explore_options = sorted(explore_options, key=lambda o: (
            game_state.board.rubble[unit_pos[0] + o[0], unit_pos[1] + o[1]]
            if 0 <= unit_pos[0] + o[0] < 48 and 0 <= unit_pos[1] + o[1] < 48 else 1000))
        if prioritise_static:  # add static option either at the beginning or at the end of the explored options
            explore_options.insert(0, (0, 0))
        else:
            explore_options.append((0, 0))

        wander_option_found = False
        for dx, dy in explore_options:
            x, y = unit_pos[0] + dx, unit_pos[1] + dy
            if not (0 <= x < 48 and 0 <= y < 48):
                continue

            # if (init_turn + i + 1, (x, y)) in position_registry.keys():
            if not is_registry_free(init_turn + i + 1, (x, y), position_registry, unit.unit_id):
                continue

            new_pos = unit_pos + np.array((dx, dy))
            distances_to_factories = chebyshev_dist_vect_point(opponent_factory_tiles, new_pos)
            if len(distances_to_factories) and np.min(distances_to_factories) == 1:
                continue  # don't go through an opponent factory

            is_adj_to_fact, dir_to_fact = adjacent_to_factory(new_pos, assigned_factory.pos)
            if (dx, dy) == (0, 0) and len(to_be_cargod) and idx_cargo < len(to_be_cargod) and is_adj_to_fact:
                actions.extend([unit.transfer(dir_to_fact, to_be_cargod[idx_cargo][1], to_be_cargod[idx_cargo][0])])
                idx_cargo += 1
            else:
                actions.extend([unit.move(direction=direction_to(unit_pos, new_pos), repeat=0, n=1)])
            actions_counter += 1
            position_registry.update({(init_turn + actions_counter, tuple(new_pos)): unit.unit_id})
            unit_pos = new_pos
            wander_option_found = True
            break

        if not wander_option_found:  # commits suicide...
            # todo: should obviously not suicide, but find best outcome based on other units around...
            #       should be uncommon enough to be fine for now
            actions.extend([unit.self_destruct()])

    # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
    #     pass

    if replace_static_by_charge and all([tuple(a) == tuple(unit.move(direction=0)) for a in actions]):
        actions = [unit.recharge(unit_cfg.BATTERY_CAPACITY)]

    return actions


def make_itinerary_advanced_old(unit, target_pos, game_state, position_registry, starting_turn=None, unit_pos=None,
                            allow_approx=True, fct_stop_condition_tile=None, max_dist=20):

    if starting_turn is None:  # assume it's a movement to be performed right now!
        starting_turn = game_state.real_env_steps

    if unit_pos is None:
        unit_pos = unit.pos

    # go somewhere closer instead...
    if manhattan_dist_points(unit_pos, target_pos) > max_dist:
        target_pos = find_intermediary_spot(unit, game_state, target_pos, unit_pos, max_dist=max_dist)
        if target_pos is None:
            return {"actions": [], "power_cost": 0, "position_registry": dict(),
                     "unit_pos": unit_pos, "actions_counter": 0}

    for max_dumb_moves in (0.5, 1.5, 2.5):

        if max_dumb_moves == 0.5 and allow_approx:  # try a full naive way for cheap first
            if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
                pass
            path, power_cost, final_tile = move_to_rec(
                unit, target_pos, game_state, position_registry, unit_pos, turn_counter=starting_turn,
                max_dumb_moves=max_dumb_moves, allow_approx=False, fct_stop_condition_tile=fct_stop_condition_tile)
            if path is not None:
                break

        # if game_state.real_env_steps in time_monitored_turns and unit.unit_id == "unit_50":
        #     # end_time = time.time()
        #     # local_time = str(end_time - start_time)
        #     # pass
        #     start_time = time.time()
        #     pass
        if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
            pass
        path, power_cost, final_tile = move_to_rec(
            unit, target_pos, game_state, position_registry, unit_pos, turn_counter=starting_turn,
            max_dumb_moves=max_dumb_moves, allow_approx=allow_approx, fct_stop_condition_tile=fct_stop_condition_tile)
        if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
            pass
        if allow_approx and path is not None and final_tile is not None:
            break
        # if game_state.real_env_steps in time_monitored_turns and unit.unit_id == "unit_50":
        #     end_time = time.time()
        #     local_time = str(end_time - start_time)
        #     pass
        #     start_time = time.time()
        # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        #     pass

        if (not allow_approx) and path is not None:
            break

    # second iteration of exploration, after early stop of the first one (hopefully much closer!)
    if allow_approx and final_tile is not None and tuple(final_tile) != tuple(target_pos):
        for max_dumb_moves in (0.5, 1.5, 2.5):
            if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
                pass
            if max_dumb_moves > 1.5 and manhattan_dist_points(unit_pos, target_pos) > 10:
                # go somewhere closer instead...
                target_pos = find_intermediary_spot(unit, game_state, target_pos, unit_pos, max_dist=5)
                if target_pos is None:
                    break
            path2, power_cost2, final_tile2 = move_to_rec(
                unit, target_pos, game_state, position_registry, np.array(final_tile),
                turn_counter=starting_turn + len(path), max_dumb_moves=max_dumb_moves, allow_approx=False,
                fct_stop_condition_tile=fct_stop_condition_tile)
            if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
                pass
            if path2 is not None:
                break
        if path2 is not None:
            path = path + path2
            power_cost = power_cost + power_cost2

    pos_registry_new, actions = dict(), list()

    # now we have to extract all the tuples in the path, to convert them to movements
    # ideally, we should group them to avoid hitting the 20 instructions limit
    if path is None:  # can't find decent path... let's waste time by staying static n turns?
        # todo: use wander_n here
        # 2 problems: 1)need assigned_factory
        # 2)n_wander should have the same signature system... can't know updated unit_pos and registry expensive to copy
        n_turns_static = 3
        actions.extend([unit.move(direction=0, repeat=0, n=n_turns_static)])
        for i in range(n_turns_static):
            pos_registry_new[(starting_turn + i + 1, tuple(unit_pos))] = unit.unit_id
        return {"actions": actions, "power_cost": 0, "position_registry": pos_registry_new,
                "unit_pos": unit_pos, "actions_counter": n_turns_static}

    i = 0
    while i < len(path):
        delta = path[i]

        n_repeated = 1  # fish for next similar deltas, update registry and count repetition
        unit_pos = unit_pos + np.array(delta)
        pos_registry_new[(starting_turn + i + n_repeated, tuple(unit_pos))] = unit.unit_id
        while i + n_repeated < len(path):
            if path[i + n_repeated] != delta:
                break
            n_repeated += 1
            unit_pos = unit_pos + np.array(delta)
            pos_registry_new[(starting_turn + i + n_repeated, tuple(unit_pos))] = unit.unit_id

        actions.extend([unit.move(direction=get_direction_code_from_delta(delta), repeat=0, n=n_repeated)])
        i += n_repeated

    return {"actions": actions, "power_cost": power_cost, "position_registry": pos_registry_new,
            "unit_pos": unit_pos, "actions_counter": len(path)}


def move_to_rec(unit, target_pos, game_state, position_registry, unit_pos, turn_counter, dumb_moves_counter=0,
                max_dumb_moves=0., step_on_resource_penalty=1.1, step_on_factory_center_penalty=1.2, allow_approx=True,
                fct_stop_condition_tile=None):
    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    # TODO: think about ICE and center of factories issues

    unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]

    # a bit silly to guess that here...
    player_me = "player_0" if unit.unit_id in game_state.units["player_0"].keys() else "player_1"
    player_op = "player_1" if player_me == "player_0" else "player_0"
    opponent_factory_tiles = np.array([f.pos for f in game_state.factories[player_op].values()])
    my_factory_tiles_as_tuples = [tuple(f.pos) for f in game_state.factories[player_me].values()]

    dx, dy = target_pos[0] - unit_pos[0], target_pos[1] - unit_pos[1]
    dx, dy = int(dx/abs(dx)) if dx else 0, int(dy/abs(dy)) if dy else 0

    if abs(dx) and abs(dy):
        explore_options = [((dx, 0), 0), ((0, dy), 0), ((0, 0), 0.5), ((-dx, 0), 1), ((0, -dy), 1)]
    elif abs(dx):
        explore_options = [((dx, 0), 0), ((0, 0), 0.5), ((0, 1), 1), ((0, -1), 1), ((-dx, 0), 1)]
    elif abs(dy):
        explore_options = [((0, dy), 0), ((0, 0), 0.5), ((1, 0), 1), ((-1, 0), 1), ((0, -dy), 1)]
    else:
        return [], 0, tuple(target_pos)  # we made it!!!!

    if fct_stop_condition_tile is not None and fct_stop_condition_tile(unit_pos):
        return [], 0, tuple(unit_pos)

    # sorted options to be explored (by dumb move quantifier then rubble)
    explore_options = sorted(
        explore_options, key=lambda o: o[1]*1000 + (
            game_state.board.rubble[unit_pos[0]+o[0][0], unit_pos[1]+o[0][1]]
            if 0 <= unit_pos[0]+o[0][0] < 48 and 0 <= unit_pos[1] + o[0][1] < 48 else 1000))

    # monitored_units, monitored_turns = ["unit_65"], [6]
    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    if tuple(unit_pos) != (5, 10):
        pass

    # # ugly hack to avoid being stuck in a silly way
    # # only reason to allow static is hope to open a path... if nothing moves around, let's not
    # allow_static = False
    # static_penalty = 0.5
    # for opt in explore_options:
    #     if opt[0] != (0, 0):
    #         in_use_at_some_point = any([
    #             not is_registry_free(turn_counter + 1 + i, opt[0], position_registry, unit.unit_id)
    #             for i in range(int(np.ceil((max_dumb_moves - dumb_moves_counter) / static_penalty)))])
    #         free_at_some_point = any([is_registry_free(turn_counter + 2 + i, opt[0], position_registry, unit.unit_id)
    #                                   for i in range(int(np.ceil((max_dumb_moves-dumb_moves_counter)/static_penalty)))])
    #         if in_use_at_some_point and free_at_some_point:  # then static makes sense
    #             allow_static = True
    #             break
    #     else:
    #         break
    #
    # if not allow_static:  # remove static
    #     explore_options = [opt for opt in explore_options if opt[0] != (0, 0)]

    last_resort_options, very_last_resort_options = list(), list()
    for opt in explore_options:
        tmp_dumb_moves_counter = dumb_moves_counter + opt[1]

        next_x, next_y = unit_pos[0] + opt[0][0], unit_pos[1] + opt[0][1]
        next_pos = np.array([next_x, next_y])

        if (not 0 <= next_x < 48) or (not 0 <= next_y < 48):
            continue  # next move not in the map

        distances_to_factories = chebyshev_dist_vect_point(opponent_factory_tiles, next_pos)
        if len(distances_to_factories) and np.min(distances_to_factories) == 1:
            continue  # don't go through an opponent factory

        # if game_state.real_env_steps in (350,) and unit.unit_id == "unit_38" and (next_x, next_y) == (3, 41):
        #     pass

        # if (turn_counter + 1, (next_x, next_y)) in position_registry.keys():
        if not is_registry_free(turn_counter + 1, (next_x, next_y), position_registry, unit.unit_id):
            continue  # this option is discarded because another of our unit intends to occupy the position

        if game_state.board.ice[next_x, next_y] or game_state.board.ore[next_x, next_y]:
            if not np.array_equal(next_pos, target_pos):
                tmp_dumb_moves_counter += step_on_resource_penalty
                # last_resort_options.append(opt)  # would strongly prefer not to step on ice or ore, it tends to be crowded
                # continue

        if (next_x, next_y) in my_factory_tiles_as_tuples:
            if not np.array_equal(next_pos, target_pos):
                tmp_dumb_moves_counter += step_on_factory_center_penalty
                # very_last_resort_options.append(opt)  # should maybe be forbidden to adventure there, units can spawn!
                # continue

        if tmp_dumb_moves_counter > max_dumb_moves:
            continue

        path, cost, final_tile = move_to_rec(unit, target_pos, game_state, position_registry, unit_pos=next_pos,
                                             turn_counter=turn_counter+1, dumb_moves_counter=tmp_dumb_moves_counter,
                                             max_dumb_moves=max_dumb_moves, allow_approx=allow_approx,
                                             fct_stop_condition_tile=fct_stop_condition_tile)
        if path is not None:
            rubble = game_state.board.rubble[next_x, next_y]
            extra_cost = int(np.floor(unit_cfg.MOVE_COST + rubble * unit_cfg.RUBBLE_MOVEMENT_COST))
            return [opt[0], *path], cost + extra_cost, final_tile

        # weird flex to still go to the closest we found
        # idea is to get close before exploring more expensive options
        # reduces running time overall!
        if path is None and allow_approx:
            return [opt[0]], cost, tuple(unit_pos)

    # # once all options have been explored, can attempt the semi discarded ones (ice / ore / factory_middle)
    # for opt in [*last_resort_options, *very_last_resort_options]:
    #     tmp_dumb_moves_counter = dumb_moves_counter + opt[1]
    #     next_x, next_y = unit_pos[0] + opt[0][0], unit_pos[1] + opt[0][1]
    #     next_pos = np.array([next_x, next_y])
    #     path, cost = move_to_rec(unit, target_pos, game_state, position_registry,
    #                              unit_pos=next_pos,
    #                              turn_counter=turn_counter+1, dumb_moves_counter=tmp_dumb_moves_counter,
    #                              max_dumb_moves=max_dumb_moves)
    #     if path is not None:
    #         rubble = game_state.board.rubble[next_x, next_y]
    #         extra_cost = int(np.floor(unit_cfg.MOVE_COST + rubble * unit_cfg.RUBBLE_MOVEMENT_COST))
    #         return [opt[0], *path], cost + extra_cost

    return None, 0, None


def move_to_rec2(unit, target_pos, game_state, position_registry, unit_pos, turn_counter, dumb_moves_counter=0,
                 max_dumb_moves=0., step_on_resource_penalty=1.1, step_on_factory_center_penalty=1.2, flex_n=3,
                 fct_stop_condition_tile=None, best_path_memory=None, cur_path=None, cur_cost=None):

    # TODO: think about ICE and center of factories issues

    unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]

    # a bit silly to guess that here...
    player_me = "player_0" if unit.unit_id in game_state.units["player_0"].keys() else "player_1"
    player_op = "player_1" if player_me == "player_0" else "player_0"
    opponent_factory_tiles = np.array([f.pos for f in game_state.factories[player_op].values()])
    my_factory_tiles_as_tuples = [tuple(f.pos) for f in game_state.factories[player_me].values()]

    cur_path = [] if cur_path is None else cur_path
    cur_cost = 0 if cur_cost is None else cur_cost
    cur_dist = manhattan_dist_points(unit_pos, target_pos)
    if best_path_memory is None:
        best_path_memory = {"best_path": [], "best_cost": 0, "best_dist": cur_dist}
    if cur_dist < best_path_memory["best_dist"]:
        best_path_memory["best_dist"] = cur_dist
        best_path_memory["best_path"] = cur_path
        best_path_memory["best_cost"] = cur_cost

    dx, dy = target_pos[0] - unit_pos[0], target_pos[1] - unit_pos[1]
    dx, dy = int(dx/abs(dx)) if dx else 0, int(dy/abs(dy)) if dy else 0

    if abs(dx) and abs(dy):
        explore_options = [((dx, 0), 0), ((0, dy), 0), ((0, 0), 0.5), ((-dx, 0), 1), ((0, -dy), 1)]
    elif abs(dx):
        explore_options = [((dx, 0), 0), ((0, 0), 0.5), ((0, 1), 1), ((0, -1), 1), ((-dx, 0), 1)]
    elif abs(dy):
        explore_options = [((0, dy), 0), ((0, 0), 0.5), ((1, 0), 1), ((-1, 0), 1), ((0, -dy), 1)]
    else:
        return cur_path, cur_cost  # we made it!!!!

    if fct_stop_condition_tile is not None and fct_stop_condition_tile(unit_pos):
        return [], 0, tuple(unit_pos)

    # sorted options to be explored (by dumb move quantifier then rubble)
    explore_options = sorted(
        explore_options, key=lambda o: o[1]*1000 + (
            game_state.board.rubble[unit_pos[0]+o[0][0], unit_pos[1]+o[0][1]]
            if 0 <= unit_pos[0]+o[0][0] < 48 and 0 <= unit_pos[1] + o[0][1] < 48 else 1000))

    if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        pass

    # # ugly hack to avoid being stuck in a silly way
    # # only reason to allow static is hope to open a path... if nothing moves around, let's not
    # allow_static = False
    # static_penalty = 0.5
    # for opt in explore_options:
    #     if opt[0] != (0, 0):
    #         in_use_at_some_point = any([
    #             not is_registry_free(turn_counter + 1 + i, opt[0], position_registry, unit.unit_id)
    #             for i in range(int(np.ceil((max_dumb_moves - dumb_moves_counter) / static_penalty)))])
    #         free_at_some_point = any([is_registry_free(turn_counter + 2 + i, opt[0], position_registry, unit.unit_id)
    #                                   for i in range(int(np.ceil((max_dumb_moves-dumb_moves_counter)/static_penalty)))])
    #         if in_use_at_some_point and free_at_some_point:  # then static makes sense
    #             allow_static = True
    #             break
    #     else:
    #         break
    #
    # if not allow_static:  # remove static
    #     explore_options = [opt for opt in explore_options if opt[0] != (0, 0)]

    for opt in explore_options:
        tmp_dumb_moves_counter = dumb_moves_counter + opt[1]

        next_x, next_y = unit_pos[0] + opt[0][0], unit_pos[1] + opt[0][1]
        next_pos = np.array([next_x, next_y])

        if (not 0 <= next_x < 48) or (not 0 <= next_y < 48):
            continue  # next move not in the map

        distances_to_factories = chebyshev_dist_vect_point(opponent_factory_tiles, next_pos)
        if len(distances_to_factories) and np.min(distances_to_factories) == 1:
            continue  # don't go through an opponent factory

        # if (turn_counter + 1, (next_x, next_y)) in position_registry.keys():
        if not is_registry_free(turn_counter + 1, (next_x, next_y), position_registry, unit.unit_id):
            continue  # this option is discarded because another of our unit intends to occupy the position

        if game_state.board.ice[next_x, next_y] or game_state.board.ore[next_x, next_y]:
            if not np.array_equal(next_pos, target_pos):
                tmp_dumb_moves_counter += step_on_resource_penalty
                # last_resort_options.append(opt)  # would strongly prefer not to step on ice or ore, it tends to be crowded
                # continue

        if (next_x, next_y) in my_factory_tiles_as_tuples:
            if not np.array_equal(next_pos, target_pos):
                tmp_dumb_moves_counter += step_on_factory_center_penalty
                # very_last_resort_options.append(opt)  # should maybe be forbidden to adventure there, units can spawn!
                # continue

        if tmp_dumb_moves_counter > max_dumb_moves:
            continue

        explored_path = [*cur_path, opt[0]]
        rubble = game_state.board.rubble[next_x, next_y]
        extra_cost = int(np.floor(unit_cfg.MOVE_COST + rubble * unit_cfg.RUBBLE_MOVEMENT_COST))
        final_path, final_cost = move_to_rec2(
            unit, target_pos, game_state, position_registry, unit_pos=next_pos, turn_counter=turn_counter+1,
            dumb_moves_counter=tmp_dumb_moves_counter, max_dumb_moves=max_dumb_moves, flex_n=flex_n,
            fct_stop_condition_tile=fct_stop_condition_tile, best_path_memory=best_path_memory, cur_path=explored_path,
            cur_cost=cur_cost + extra_cost)

        if final_path is not None:
            return final_path, final_cost

        # weird flex to still go to the closest we found
        # idea is to get close before exploring more expensive options
        # reduces running time overall!
        if final_path is None and flex_n is not None:
            # best_dist_so_far, best_path_so_far = best_itin_so_far
            if best_path_memory["best_dist"] + flex_n <= cur_dist:
                return best_path_memory["best_path"], best_path_memory["best_cost"]

    return None, 0


def make_itinerary_advanced(unit, target_pos, game_state, position_registry, starting_turn=None, unit_pos=None,
                                fct_stop_condition_tile=None, flex_n=3, max_dist=20):

    if starting_turn is None:  # assume it's a movement to be performed right now!
        starting_turn = game_state.real_env_steps

    if unit_pos is None:
        unit_pos = unit.pos

    # # go somewhere closer instead...
    # if manhattan_dist_points(unit_pos, target_pos) > max_dist:
    #     target_pos = find_intermediary_spot(unit, game_state, target_pos, unit_pos, max_dist=max_dist)
    #     if target_pos is None:
    #         return {"actions": [], "power_cost": 0, "position_registry": dict(),
    #                  "unit_pos": unit_pos, "actions_counter": 0}

    # for max_dumb_moves in (0.5, 1.5, 2.5):
        # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        #     pass
    path, power_cost = move_to_rec2(
        unit, target_pos, game_state, position_registry, unit_pos, turn_counter=starting_turn,
        max_dumb_moves=2.5, fct_stop_condition_tile=fct_stop_condition_tile, flex_n=flex_n)
        # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        #     pass
    pos_registry_new, actions = dict(), list()

    # now we have to extract all the tuples in the path, to convert them to movements
    # ideally, we should group them to avoid hitting the 20 instructions limit
    if path is None:  # can't find decent path... let's waste time by staying static n turns?
        # todo: use wander_n here
        # 2 problems: 1)need assigned_factory
        # 2)n_wander should have the same signature system... can't know updated unit_pos and registry expensive to copy
        n_turns_static = 3
        actions.extend([unit.move(direction=0, repeat=0, n=n_turns_static)])
        for i in range(n_turns_static):
            pos_registry_new[(starting_turn + i + 1, tuple(unit_pos))] = unit.unit_id
        return {"actions": actions, "power_cost": 0, "position_registry": pos_registry_new,
                "unit_pos": unit_pos, "actions_counter": n_turns_static}

    i = 0
    while i < len(path):
        delta = path[i]

        n_repeated = 1  # fish for next similar deltas, update registry and count repetition
        unit_pos = unit_pos + np.array(delta)
        pos_registry_new[(starting_turn + i + n_repeated, tuple(unit_pos))] = unit.unit_id
        while i + n_repeated < len(path):
            if path[i + n_repeated] != delta:
                break
            n_repeated += 1
            unit_pos = unit_pos + np.array(delta)
            pos_registry_new[(starting_turn + i + n_repeated, tuple(unit_pos))] = unit.unit_id

        actions.extend([unit.move(direction=get_direction_code_from_delta(delta), repeat=0, n=n_repeated)])
        i += n_repeated

    return {"actions": actions, "power_cost": power_cost, "position_registry": pos_registry_new,
            "unit_pos": unit_pos, "actions_counter": len(path)}