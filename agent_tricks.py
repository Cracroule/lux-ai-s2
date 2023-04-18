import collections
import numpy as np
import time
import scipy
import sys
from lux.utils_raoul import get_direction_code_from_delta, manhattan_dist_vect_point, manhattan_dist_points, \
    adjacent_to_factory, nearest_factory_tile, chebyshev_dist_vect_point, chebyshev_dist_points, \
    score_rubble_tiles_to_dig, score_rubble_add_proximity_penalty_to_tiles_to_dig, custom_dist_points, \
    is_registry_free, count_day_turns, is_unit_stronger, get_pos_power_cargo
# from scipy.spatial.distance import cdist

from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory


# debug / analysis purposes
time_monitored_turns = []
monitored_turns = [425, 426, 427, 428, 429, 430, 431]
monitored_units = ["unit_99"]


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

    # go to tile
    # # todo: make itinerary computation before power (but consequences after), so we can use exact power computation
    # #       instead of proxy
    itin_d = make_itinerary_advanced(unit, target_dug_tile, game_state, position_registry,
                                     starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
    actions.extend(itin_d["actions"])
    actions_counter += itin_d["actions_counter"]
    position_registry.update(itin_d["position_registry"])
    unit_pos, power_cost = itin_d["unit_pos"], itin_d["power_cost"]

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

    if unit.unit_id == "unit_99" and init_turn == 426:
        pass

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

    # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
    #     pass

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


def go_fight(unit, game_state, position_registry, op_unit, assigned_factory, threat_desc, allow_weaker=False):
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

#
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


def make_itinerary_advanced(unit, target_pos, game_state, position_registry, starting_turn=None, unit_pos=None,
                            allow_approx=True):

    if starting_turn is None:  # assume it's a movement to be performed right now!
        starting_turn = game_state.real_env_steps

    if unit_pos is None:
        unit_pos = unit.pos

    for max_dumb_moves in (0.5, 1.5, 2.5):

        # if game_state.real_env_steps in time_monitored_turns and unit.unit_id == "unit_50":
        #     # end_time = time.time()
        #     # local_time = str(end_time - start_time)
        #     # pass
        #     start_time = time.time()
        #     pass
        # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        #     pass
        path, power_cost, final_tile = move_to_rec(unit, target_pos, game_state, position_registry, unit_pos,
                                                   turn_counter=starting_turn, max_dumb_moves=max_dumb_moves,
                                                   allow_approx=allow_approx)
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
            path2, power_cost2, final_tile2 = move_to_rec(unit, target_pos, game_state, position_registry,
                                                          np.array(final_tile),
                                                          turn_counter=starting_turn + len(path),
                                                          max_dumb_moves=max_dumb_moves, allow_approx=False)
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
                max_dumb_moves=0., step_on_resource_penalty=1.1, step_on_factory_center_penalty=1.2, allow_approx=True):

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

    # sorted options to be explored (by dumb move quantifier then rubble)
    explore_options = sorted(
        explore_options, key=lambda o: o[1]*1000 + (
            game_state.board.rubble[unit_pos[0]+o[0][0], unit_pos[1]+o[0][1]]
            if 0 <= unit_pos[0]+o[0][0] < 48 and 0 <= unit_pos[1] + o[0][1] < 48 else 1000))

    # monitored_units, monitored_turns = ["unit_65"], [6]
    # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
    #     pass

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

        path, cost, final_tile = move_to_rec(unit, target_pos, game_state, position_registry,
                                             unit_pos=next_pos,
                                             turn_counter=turn_counter+1, dumb_moves_counter=tmp_dumb_moves_counter,
                                             max_dumb_moves=max_dumb_moves)
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
