import collections
import numpy as np
import time
import scipy
import sys
from lux.utils_raoul import get_direction_code_from_delta, manhattan_dist_vect_point, manhattan_dist_points, \
    adjacent_to_factory, nearest_factory_tile, chebyshev_dist_vect_point, chebyshev_distance_points, \
    score_rubble_tiles_to_dig, score_rubble_add_proximity_penalty_to_tiles_to_dig, custom_dist_points, \
    is_registry_free, count_day_turns, is_unit_stronger
# from scipy.spatial.distance import cdist

from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory


# TODO: FOR ALL (Especially solo diggers): IF LOW ENERGY, CHARGE (ELSE CAN END UP BEING STUCK ON LOW ENERGY...)
# TODO: COLLISIONS BECAUSE OF POWER/DIG STEALING THE TILE SOMEONE COUNTED ON
# TODO: DETECT ENEMY, MOVE DIFFERENTLY


# def go_take_power(unit, game_state, position_registry, target_dug_tile, target_factory):
#     pass

# def make_dig_itinerary(unit, target_dug_tile, target_factory, game_state, n_dig=None, ideal_n_digs=15, min_n_digs=7,
#                        resource_type=0):
#     """
#     returns a full list of actions:
#     (take power) > move to tile > dig to get resource > go back to the nearest factory > transfer resource
#     if n_dig is None, then will do as many digs as possible, at least min_n_digs, at most ideal_n_digs
#     if not enough power of at least min_n_digs, will try to suck out power to achieve ideal_n_digs
#
#     :param unit:
#     :param target_dug_tile:
#     :param target_factory:
#     :param game_state:
#     :param n_dig:
#     :param ideal_n_digs:
#     :param min_n_digs:
#     :param resource_type: 0 for water
#     :return:
#     """
#     actions = list()
#     unit_pos = unit.pos
#     unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
#
#     near_factory_tile_from_dig_spot = nearest_factory_tile(target_dug_tile, target_factory.pos)
#
#     if np.array_equal(unit_pos, target_factory.pos):
#         actions.extend(make_itinerary(unit, target_pos=near_factory_tile_from_dig_spot, unit_pos=unit_pos))
#         unit_pos = near_factory_tile_from_dig_spot
#
#     if n_dig is None:
#         dist = manhattan_dist_points(unit_pos, target_dug_tile)
#         nb_rubble = 50  # proxy rubble for simplicity, should check actual rubble on map
#         power_cost_move = 2 * dist * (unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * nb_rubble)
#         n_dig = int(np.floor((unit.power - power_cost_move) / unit_cfg.DIG_COST))
#
#         if n_dig < min_n_digs:  # not enough energy, let's refill first
#             near_factory_tile = nearest_factory_tile(unit_pos, target_factory.pos)
#             actions.extend(make_itinerary(unit, target_pos=near_factory_tile, unit_pos=unit_pos))
#             unit_pos = near_factory_tile
#             sucked_power = max(min(unit_cfg.BATTERY_CAPACITY - unit.power,
#                                    ideal_n_digs * unit_cfg.DIG_COST + power_cost_move - unit.power),
#                                target_factory.power)
#             actions.extend([unit.pickup(4,  sucked_power)])
#             n_dig = int(np.floor((unit.power + sucked_power - power_cost_move) / unit_cfg.DIG_COST))
#
#     actions.extend(make_itinerary(unit, target_pos=target_dug_tile, unit_pos=unit_pos))
#     actions.extend([unit.dig(repeat=0, n=n_dig)])
#
#     n_dig_rubble = int(np.ceil(game_state.board.rubble[target_dug_tile[0], target_dug_tile[1]] /
#                                unit_cfg.DIG_RUBBLE_REMOVED))
#     expected_ice_gain = (n_dig - n_dig_rubble) * unit_cfg.DIG_RESOURCE_GAIN
#
#     near_factory_tile = nearest_factory_tile(target_dug_tile, target_factory.pos)
#     actions.extend(make_itinerary(unit, target_pos=near_factory_tile, unit_pos=target_dug_tile))
#     # resource_type = 0  # water
#     actions.extend([unit.transfer(0, resource_type, expected_ice_gain, repeat=0)])
#     return actions


def go_adj_dig(unit, game_state, position_registry, target_dug_tile, target_factory, factories_power, n_min_digs=4,
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
    near_factory_tile_from_dig_spot = nearest_factory_tile(target_dug_tile, target_factory.pos)
    assert manhattan_dist_points(near_factory_tile_from_dig_spot, target_dug_tile) == 1

    actions, actions_counter = list(), 0
    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    unit_pos = unit.pos

    # if nearby factory AND carrying resources transfer them (reassignment handling) and stop (will take power after)
    is_adj, dir_to_fac = adjacent_to_factory(unit_pos, target_factory.pos)
    for cargo_amt, r_code in [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]:
        if is_adj and cargo_amt:
            actions.extend([unit.transfer(dir_to_fac, r_code, cargo_amt)])
            actions_counter += 1
            position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})
            return actions

    # # itinerary computation made before, so we know how much it costs to get there
    # itin_d = make_itinerary_advanced(unit, target_dug_tile, game_state, position_registry,
    #                                  starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
    # unit_pos_after_moving, power_cost = itin_d["unit_pos"], itin_d["power_cost"]

    # pick power?
    min_power = unit_cfg.DIG_COST * n_min_digs  # 5 * 60 = 300 for heavy
    desired_power = unit_cfg.DIG_COST * n_desired_digs  # 8 * 60 = 480 for heavy
    if unit.power < desired_power:
        if factories_power[target_factory.unit_id] >= desired_power - unit.power:
            picked_power = desired_power - unit.power
        elif unit.power > min_power:
            picked_power = 0  # let's not pick if we have a reasonable amount
        elif factories_power[target_factory.unit_id] >= min_power - unit.power:
            picked_power = min_power - unit.power
        else:
            n_turns_static = 3
            actions.extend([unit.move(direction=0, repeat=0, n=n_turns_static)])
            for i in range(n_turns_static):
                position_registry[(init_turn + i + 1, tuple(unit_pos))] = unit.unit_id
            return actions  # abort and sacrifice turn by waiting... should be handled more elegantly
        if picked_power:
            if chebyshev_distance_points(target_factory.pos, unit_pos) > 1:  # can't pick power if not on a factory tile
                # todo: should check if we have an assistant, if not go back to base...
                n_turns_static = 3
                actions.extend([unit.move(direction=0, repeat=0, n=n_turns_static)])
                for i in range(n_turns_static):
                    position_registry[(init_turn + i + 1, tuple(unit_pos))] = unit.unit_id
                return actions
            actions.extend([unit.pickup(4, picked_power)])
            actions_counter += 1
            position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

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


def assist_adj_digging(unit, game_state, position_registry, assisted_unit, target_factory, factories_power,
                       target_dug_tile=None, n_desired_digs=5):
    """
    - [pick power] (48 for heavy assisted_unit) (repeat)
    - go to the nearest tile within factory
    - transfer power to assisted_unit (repeat)

    # important to start by pick to prevent over picking power which makes game crash)

    :return: [actions]
    """

    actions, actions_counter = list(), 0
    unit_pos = unit.pos

    # ignore below for now
    # # then the unit is not on the factory! we bring it back to the factory and that's it, can't take power in the future
    # if manhattan_dist_points(target_factory.pos, unit.pos) > 1:
    #     return actions.extend(go_back_to_factory(unit, game_state, agent, target_factory))

    if target_dug_tile is None:
        target_dug_tile = assisted_unit.pos

    # make sure we're only dealing with adjacent-to-factory dug tiles
    near_factory_tile_from_dig_spot = nearest_factory_tile(target_dug_tile, target_factory.pos)
    assert manhattan_dist_points(near_factory_tile_from_dig_spot, target_dug_tile) == 1

    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    assist_unit_cfg = game_state.env_cfg.ROBOTS[assisted_unit.unit_type]

    # move nearby dedicated spot to be able to transfer stuff, then wait for instructions
    if not np.array_equal(unit_pos, near_factory_tile_from_dig_spot):
        itin_d = make_itinerary_advanced(unit, near_factory_tile_from_dig_spot, game_state, position_registry,
                                         starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
        actions.extend(itin_d["actions"])
        position_registry.update(itin_d["position_registry"])
        # actions_counter += itin_d["actions_counter"]
        # unit_pos, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
        return actions

    # if tuple(unit_pos) == (1, 43):
    #     a = [(t, (x, y)) for (t, (x, y)) in position_registry.keys() if tuple((x, y)) == (1, 43)]
    #     pass

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

    if factories_power[target_factory.unit_id] > 1000 and assisted_unit.power < 0.8 * assist_unit_cfg.BATTERY_CAPACITY:
        sucked_power += int(assist_unit_cfg.BATTERY_CAPACITY / 30)  # can take much more, factory has plenty

    sucked_power = min((sucked_power, factories_power[target_factory.unit_id],
                        unit_cfg.BATTERY_CAPACITY - unit.power,
                        assist_unit_cfg.BATTERY_CAPACITY - assisted_unit.power))

    # need to make a difference between sucked power and transmitted power, as assistant might be full
    transmitted_power = min(unit.power + sucked_power - 5, int(0.9 * (unit.power + sucked_power)))

    # take corresponding power
    actions.extend([unit.pickup(4, sucked_power, repeat=0)])
    factories_power[target_factory.unit_id] -= sucked_power
    # actions_counter += 1
    # position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    # transfer power
    resource_type = 4  # power
    transfer_direction = direction_to(near_factory_tile_from_dig_spot, target_dug_tile)
    actions.extend([unit.transfer(transfer_direction, resource_type, transmitted_power, repeat=0)])
    # actions_counter += 1
    # position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    return actions


def go_to_factory(unit, game_state, position_registry, target_factory, unit_pos=None, tiles_for_assistants_only=None,
                  starting_turn=None, assume_stay_extra_turn=True, anticipated_power=None, anticipated_cargo=None,
                  can_reassign=False):
    """
    - go back to the nearest tile within factory
    - [drop stuff being carried]

    (that's it... Idea is to call that one if running out of power when digging for example...)

    :return: [actions]
    """

    if anticipated_cargo is None:
        anticipated_cargo = [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]

    if anticipated_power is None:
        anticipated_power = unit.power

    actions, actions_counter = list(), 0
    unit_pos = unit.pos if unit_pos is None else unit_pos

    unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
    init_turn = game_state.real_env_steps if starting_turn is None else starting_turn

    cargo_counter = 0
    for cargo_amt, r_code in anticipated_cargo:
        cargo_counter += 1 if cargo_amt else 0
        # todo: if adjacent to factory, transfer cargo to factory already

    # return {"actions": actions, "power_cost": power_cost, "position_registry": pos_registry_new,
    #         "unit_pos": unit_pos, "actions_counter": len(path)}

    excluded_tiles = tiles_for_assistants_only
    while len(excluded_tiles) < 8:
        near_factory_tile = nearest_factory_tile(unit_pos, target_factory.pos, excluded_tiles=excluded_tiles)

        # monitored_turns = []
        # if game_state.real_env_steps in monitored_turns and unit.unit_id == "unit_50":
        #     pass
        #     start_time = time.time()

        manh_dist_to_tile = manhattan_dist_points(near_factory_tile, unit_pos)
        tile_stay_length = cargo_counter + (1 if assume_stay_extra_turn else 0) + 1
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

        if power_cost + unit_cfg.ACTION_QUEUE_POWER_COST >= anticipated_power:  # unless not enough power (recharge if so)
            # todo: enquire if enough space to stay around before attempting to charge... Will block way for some robots
            actions.extend([unit.recharge(power_cost + unit_cfg.ACTION_QUEUE_POWER_COST)])
            return actions
            # excluded_tiles.append(tuple(near_factory_tile))
            # continue

        if np.array_equal(candidate_unit_pos, near_factory_tile) and all(
                [is_registry_free(init_turn + actions_counter + itin_d["actions_counter"] + i + 1,
                                  near_factory_tile, position_registry, unit.unit_id) for i in
                 range(cargo_counter + (1 if assume_stay_extra_turn else 0))]):
            actions.extend(itin_d["actions"])
            position_registry.update(itin_d["position_registry"])
            actions_counter += itin_d["actions_counter"]
            unit_pos = candidate_unit_pos
            break
            # return actions

        # if it didn't work, try to make it work with next tile...
        excluded_tiles.append(tuple(near_factory_tile))

    if len(actions) == 0:  # we failed to find a way to unload cargo and to sit around for one turn
        n_wander = 3
        actions.extend(wander_n_turns(unit, game_state, position_registry, target_factory, n_wander=n_wander,
                                      unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
        actions_counter += 3
        return actions

    # if unit.unit_id == "unit_50" and game_state.real_env_steps == 866:
    #     end_time = time.time()
    #     local_time = str(end_time - start_time)
    #     pass
    #     start_time = time.time()

    # transfer carried resources
    for cargo_amt, r_code in anticipated_cargo:
        if cargo_amt:
            actions.extend([unit.transfer(0, r_code, cargo_amt)])
            actions_counter += 1
            position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    # cheeky extra book to take power... (but power NOT taken, assumed to be first action next)
    position_registry.update({(init_turn + actions_counter + 1, tuple(unit_pos)): unit.unit_id})

    return actions


def wander_n_turns(unit, game_state, position_registry, target_factory, n_wander=3, unit_pos=None, starting_turn=None):
    # function to stick around and trying to survive when nothing seems to work (i.e. wait for n turns)
    # todo: if move of +dx, next favored move should be -dx...
    # TODO: unload cargo ONLY if nearby factory (did i forget this ?)
    unit_pos = unit.pos if unit_pos is None else unit_pos
    init_turn = game_state.real_env_steps if starting_turn is None else starting_turn
    actions, actions_counter = list(), 0

    cargo_options = [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]
    to_be_cargod = [(cargo_amt, r_code) for cargo_amt, r_code in cargo_options
                    if cargo_amt and adjacent_to_factory(unit_pos, target_factory.pos)]

    idx_cargo = 0
    for i in range(n_wander):
        if is_registry_free(init_turn + i + 1, unit_pos, position_registry, unit.unit_id):
            if len(to_be_cargod) and idx_cargo < len(to_be_cargod):
                actions_counter += 1
                actions.extend([unit.transfer(0, to_be_cargod[idx_cargo][1], to_be_cargod[idx_cargo][0])])
                position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})
                idx_cargo += 1
            else:
                actions.extend([unit.move(direction=0, repeat=0, n=1)])
        else:
            explore_options = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            explore_options = sorted(explore_options, key=lambda o: (
                game_state.board.rubble[unit_pos[0] + o[0], unit_pos[1] + o[1]]
                if 0 <= unit_pos[0] + o[0] < 48 and 0 <= unit_pos[1] + o[1] < 48 else 1000))

            wander_option_found = False
            for dx, dy in explore_options:
                x, y = unit_pos[0] + dx, unit_pos[1] + dy
                if not (0 <= x < 48 and 0 <= y < 48):
                    continue
                # if (init_turn + i + 1, (x, y)) in position_registry.keys():
                if not is_registry_free(init_turn + i + 1, (x, y), position_registry, unit.unit_id):
                    continue
                new_pos = unit_pos + np.array((x, y))
                actions_counter += 1
                actions.extend([unit.move(direction=direction_to(unit_pos, new_pos), repeat=0, n=1)])
                position_registry.update({(init_turn + actions_counter, tuple(new_pos)): unit.unit_id})
                unit_pos = new_pos
                wander_option_found = True
                break

            if not wander_option_found:  # commits suicide...
                # todo: should obviously not suicide, but find best outcome based on other units around...
                #       should be uncommon enough to be fine for now
                actions.extend([unit.self_destruct()])
    return actions


# to be used on ice or ore
def go_dig_resource(unit, game_state, position_registry, target_dug_tile, target_factory, factories_power, n_min_digs=5,
                    n_desired_digs=8, tiles_for_assistants_only=None):
    """
    - [move if in the middle of the factory and stop actions there]
    - [pick power]
    - go to tile
    - dig
    - come back nearby factory
    - transfer

    # important to start with power to prevent over picking power which makes game crash)
    :return: [actions]
    """

    # TODO: RECOVERY MANAGEMENT: IF HAVE SOME RESOURCES AND NEARBY FACTORY, DROP THEM!!
    #       ACTUALLY ALL THE ROLES SHOULD START LIKE THAT (except adj-tasks obviously)

    # todo: MASSIVE ISSUE: all the position_registry.update that are NOT following a make_itinerary are actually
    #       not checking for availability of the tile, they just take. They should be able to compromise if
    #       another robot intends to use the tile...
    #   two options:
    #     -implement booking space (with potential move around waiting for spot to be free, or to find alternative spot)
    #     OR    - implement collision prevention, i.e. stop least urgent before collision

    actions, actions_counter = list(), 0

    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    unit_pos, power = unit.pos, unit.power

    nearest_factory_tile_from_dug_tile = nearest_factory_tile(target_dug_tile, target_factory.pos,
                                                              excluded_tiles=tiles_for_assistants_only)
    if np.array_equal(unit_pos, target_factory.pos):  # if it is on the central factory tile, move (unsafe)
        itin_d = make_itinerary_advanced(unit, nearest_factory_tile_from_dug_tile, game_state, position_registry,
                                         starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
        actions.extend(itin_d["actions"])
        position_registry.update(itin_d["position_registry"])
        # actions_counter += itin_d["actions_counter"]
        # unit_pos, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
        return actions

    is_ice_dig = game_state.board.ice[target_dug_tile[0], target_dug_tile[1]]
    is_ore_dig = game_state.board.ore[target_dug_tile[0], target_dug_tile[1]]
    assert is_ice_dig or is_ore_dig

    # if nearby factory AND carrying resources transfer them (reassignment handling) and stop (will take power after)
    is_adj, dir_to_fac = adjacent_to_factory(unit_pos, target_factory.pos)
    for cargo_amt, r_code in [(unit.cargo.ice, 0), (unit.cargo.ore, 1), (unit.cargo.water, 2), (unit.cargo.metal, 3)]:
        if is_adj and cargo_amt:
            actions.extend([unit.transfer(dir_to_fac, r_code, cargo_amt)])
            actions_counter += 1
            position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})
            return actions

    # dist = manhattan_dist_points(unit_pos, target_dug_tile)
    # nb_rubble = 50  # proxy rubble for simplicity, should check actual rubble on map
    # power_cost_move = int(2 * dist * (unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * nb_rubble))
    # desired_buffer = int(unit_cfg.BATTERY_CAPACITY / 30)

    # # itinerary computation made before, so we know how much it costs to get there
    # TODO: care: it's assumed to be the same cost to come back to the factory (x2 factor ), which might be VERY wrong\
    # TODO: could use proxy to compute the cost... can't use itinerary, because we might take power, which
    #       f*cks up the position registry because we're then late by one turn
    itin_d = make_itinerary_advanced(unit, target_dug_tile, game_state, position_registry,
                                     starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
    unit_pos_after_moving, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
    power_cost_move = 2 * power_cost
    desired_buffer = int(unit_cfg.BATTERY_CAPACITY / 30)

    # pick power?
    min_power = min(unit_cfg.DIG_COST * n_min_digs + power_cost_move,
                    unit_cfg.BATTERY_CAPACITY)  # 5 * 60 = 300 for heavy
    desired_power = min(unit_cfg.DIG_COST * n_desired_digs + power_cost_move + desired_buffer,
                        unit_cfg.BATTERY_CAPACITY)  # 8 * 60 = 480 for heavy
    picked_power = 0
    cur_cheb_dist_to_factory = chebyshev_distance_points(target_factory.pos, unit_pos)
    if unit.power < desired_power:
        if factories_power[target_factory.unit_id] >= desired_power - unit.power:
            picked_power = desired_power - unit.power
        elif unit.power > min_power:
            picked_power = 0  # let's not pick if we have a reasonable amount
        elif factories_power[target_factory.unit_id] >= min_power - unit.power:
            # picked_power = min_power - unit.power
            picked_power = factories_power[target_factory.unit_id]
        elif cur_cheb_dist_to_factory <= 1:
            # # todo: find something better to do when no energy...
            # #      charging is good, but then it should be caught to be reset once it's possible to do better
            # #      i.e. here when the factory has power...
            n_wander = 3
            actions.extend(wander_n_turns(unit, game_state, position_registry, target_factory, n_wander=n_wander,
                                          unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
            actions_counter += 3
            return actions

        if cur_cheb_dist_to_factory > 1 and (unit.power + picked_power < min_power or picked_power):
            actions.extend(go_to_factory(
                unit, game_state, position_registry, target_factory, unit_pos=unit_pos,
                tiles_for_assistants_only=tiles_for_assistants_only, starting_turn=init_turn + actions_counter))
            return actions

        if picked_power:
            if factories_power[target_factory.unit_id] > 1000:  # can take much more, factory has plenty
                picked_power = min(unit_cfg.BATTERY_CAPACITY - unit.power,
                                   max(picked_power, int(0.10 * (factories_power[target_factory.unit_id] - 900))))

            if is_registry_free(init_turn + actions_counter + 1, unit_pos, position_registry, unit.unit_id):
                actions.extend([unit.pickup(4, picked_power)])
                position_registry.update({(init_turn + actions_counter + 1, tuple(unit_pos)): unit.unit_id})
                actions_counter += 1
                factories_power[target_factory.unit_id] -= picked_power
                power += picked_power
            else:
                actions.extend(wander_n_turns(unit, game_state, position_registry, target_factory,
                                              n_wander=2, unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
                return actions

    # actually go to target
    itin_d = make_itinerary_advanced(unit, target_dug_tile, game_state, position_registry,
                                     starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
    unit_pos, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
    actions.extend(itin_d["actions"])
    position_registry.update(itin_d["position_registry"])
    actions_counter += itin_d["actions_counter"]
    power -= power_cost
    if not np.array_equal(unit_pos, target_dug_tile):
        return actions  # could not find a way... abort other steps for now

    # dig
    # todo: actually check if we'll have resources after digging, if not anticipated_cargo should be zero
    n_dig_eventually = int(np.floor(0.9 * (unit.power + picked_power - power_cost_move) / unit_cfg.DIG_COST))
    actions.extend([unit.dig(repeat=0, n=n_dig_eventually)])
    power -= unit_cfg.DIG_COST * n_dig_eventually
    for i in range(n_dig_eventually):
        actions_counter += 1
        position_registry.update({(init_turn + actions_counter, tuple(unit_pos)): unit.unit_id})

    power += count_day_turns(
        turn_start=init_turn, turn_end=init_turn + actions_counter) * unit_cfg.CHARGE + unit_cfg.ACTION_QUEUE_POWER_COST
    anticipated_cargo = [(unit.cargo.ice + is_ice_dig * n_dig_eventually * unit_cfg.DIG_RESOURCE_GAIN, 0),
                         (unit.cargo.ore + is_ore_dig * n_dig_eventually * unit_cfg.DIG_RESOURCE_GAIN, 1),
                         (unit.cargo.water, 2), (unit.cargo.metal, 3)]
    actions.extend(go_to_factory(unit, game_state, position_registry, target_factory, unit_pos,
                                 tiles_for_assistants_only, starting_turn=init_turn + actions_counter,
                                 anticipated_power=power, anticipated_cargo=anticipated_cargo))

    return actions


def go_dig_rubble(unit, game_state, position_registry, assigned_factory, factories_power, rubble_tiles_being_dug,
                  tiles_scores=None, n_min_digs=5, n_desired_digs=8, tiles_for_assistants_only=None):

    obs_n_tiles = 10
    if tiles_scores is None:
        tiles_scores = score_rubble_tiles_to_dig(game_state, assigned_factory, obs_n=obs_n_tiles)

    actions, actions_counter = list(), 0
    unit_cfg, init_turn = game_state.env_cfg.ROBOTS[unit.unit_type], game_state.real_env_steps
    unit_pos = unit.pos

    # change score metric to slightly favor tiles to dug nearby unit_pos
    tiles_scores = score_rubble_add_proximity_penalty_to_tiles_to_dig(tiles_scores, unit_pos)
    tiles_scores = {t: sc for t, sc in tiles_scores.items() if tuple(t) not in rubble_tiles_being_dug.values()}
    if not len(tiles_scores):
        tiles_scores = score_rubble_tiles_to_dig(game_state, assigned_factory, obs_n=14)
        tiles_scores = score_rubble_add_proximity_penalty_to_tiles_to_dig(tiles_scores, unit_pos)
        tiles_scores = {t: sc for t, sc in tiles_scores.items() if tuple(t) not in rubble_tiles_being_dug.values()}
        if not len(tiles_scores):
            actions.extend([unit.recharge(unit_cfg.BATTERY_CAPACITY)])
            # all rubble has been dug... should do something else...
            return actions

    ordered_tiles = np.array(sorted(tiles_scores.keys(), key=lambda t: tiles_scores[(t[0], t[1])]))
    best_candidate_target_dug_tile = ordered_tiles[0]  # need one only to move away if in center of factory

    nearest_factory_tile_from_dug_tile = nearest_factory_tile(best_candidate_target_dug_tile, assigned_factory.pos,
                                                              excluded_tiles=tiles_for_assistants_only)
    if np.array_equal(unit_pos, assigned_factory.pos):  # if it is on the central factory tile, move (unsafe)
        itin_d = make_itinerary_advanced(unit, nearest_factory_tile_from_dug_tile, game_state,
                                         position_registry, starting_turn=init_turn + actions_counter,
                                         unit_pos=unit_pos)
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
    nb_rubble = 50  # proxy rubble for simplicity, should check actual rubble on map
    power_cost_move = int((dist + dist2) * (unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * nb_rubble))
    desired_buffer = int(unit_cfg.BATTERY_CAPACITY / 30)

    if game_state.real_env_steps >= 99 and unit.unit_id == "unit_38":
        pass

    # # itinerary computation made before, so we know how much it costs to get there
    # TODO: care: it's assumed to be the same cost to come back to the factory (x2 factor), which might be VERY wrong
    # itin_d = make_itinerary_advanced(unit, target_dug_tile, game_state, position_registry,
    #                                  starting_turn=init_turn + actions_counter, unit_pos=unit_pos)
    # unit_pos_after_moving, power_cost = itin_d["unit_pos"], itin_d["power_cost"]
    # power_cost_move = 2 * power_cost
    # desired_buffer = int(unit_cfg.BATTERY_CAPACITY / 30)

    # n_desired_digs = min(n_desired_digs, n_necessary_digs)
    # n_min_digs = min(n_desired_digs, n_min_digs)

    # pick power?
    min_power = min(unit_cfg.DIG_COST * n_min_digs + power_cost_move,
                    unit_cfg.BATTERY_CAPACITY)  # 5 * 60 = 300 for heavy
    desired_power = min(unit_cfg.DIG_COST * n_desired_digs + power_cost_move + desired_buffer,
                        unit_cfg.BATTERY_CAPACITY)  # 8 * 60 = 480 for heavy
    picked_power = 0
    cur_cheb_dist_to_factory = chebyshev_distance_points(assigned_factory.pos, unit_pos)
    if unit.power < desired_power:
        if factories_power[assigned_factory.unit_id] >= desired_power - unit.power:
            picked_power = desired_power - unit.power
        elif unit.power > min_power:
            picked_power = 0  # let's not pick if we have a reasonable amount
        elif factories_power[assigned_factory.unit_id] >= min_power - unit.power:
            picked_power = min_power - unit.power
            # todo: uncomment below, comment top ?
            # picked_power = factories_power[assigned_factory.unit_id]
        elif cur_cheb_dist_to_factory <= 1:  # if on factory tile and factory poor, we wait
            n_wander = 3
            actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory, n_wander=n_wander,
                                          unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
            actions_counter += 3
            return actions

    if cur_cheb_dist_to_factory > 1 and (unit.power + picked_power < min_power or picked_power):
        actions.extend(go_to_factory(unit, game_state, position_registry, assigned_factory, unit_pos,
                                     tiles_for_assistants_only, starting_turn=init_turn + actions_counter))
        return actions

    if picked_power:
        if factories_power[assigned_factory.unit_id] > 1000:  # can take much more, factory has plenty
            picked_power = min(unit_cfg.BATTERY_CAPACITY - unit.power,
                               max(picked_power, int(0.10 * (factories_power[assigned_factory.unit_id] - 900))))

        if is_registry_free(init_turn + actions_counter + 1, unit_pos, position_registry, unit.unit_id):
            actions.extend([unit.pickup(4, picked_power)])
            position_registry.update({(init_turn + actions_counter + 1, tuple(unit_pos)): unit.unit_id})
            actions_counter += 1
            factories_power[assigned_factory.unit_id] -= picked_power
        else:
            actions.extend(wander_n_turns(unit, game_state, position_registry, assigned_factory,
                                          n_wander=2, unit_pos=unit_pos, starting_turn=init_turn + actions_counter))
            return actions

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
    rubble_tiles_being_dug[unit.unit_id] = tuple(target_dug_tile)
    position_registry.update(itin_d["position_registry"])
    actions_counter += itin_d["actions_counter"]
    # if not np.array_equal(unit_pos, target_dug_tile):
    #     return actions  # could not find a way... abort other steps for now

    # dig a reasonable amount of times (function of power and quantity of rubble)
    n_necessary_digs = int(np.ceil(
        game_state.board.rubble[target_dug_tile[0], target_dug_tile[1]] / unit_cfg.DIG_RUBBLE_REMOVED))
    n_dig_eventually = min(tile_available_counter, min(int(np.floor(
        (unit.power + picked_power - power_cost_move) / unit_cfg.DIG_COST)), n_necessary_digs))
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
                                  starting_turn=init_turn + actions_counter))
    return actions


def make_itinerary_advanced(unit, target_pos, game_state, position_registry, starting_turn=None, unit_pos=None):

    if starting_turn is None:  # assume it's a movement to be performed right now!
        starting_turn = game_state.real_env_steps

    if unit_pos is None:
        unit_pos = unit.pos

    for max_dumb_moves in (0.5, 1.5, 2.5):
        # time_monitored_turns = []
        # monitored_turns = [6]
        # monitored_units = ["unit_65"]

        # if game_state.real_env_steps in time_monitored_turns and unit.unit_id == "unit_50":
        #     # end_time = time.time()
        #     # local_time = str(end_time - start_time)
        #     # pass
        #     start_time = time.time()
        #     pass

        # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        #     pass

        path, power_cost = move_to_rec(unit, target_pos, game_state, position_registry, unit_pos,
                                       turn_counter=starting_turn, max_dumb_moves=max_dumb_moves)
        # if game_state.real_env_steps in time_monitored_turns and unit.unit_id == "unit_50":
        #     end_time = time.time()
        #     local_time = str(end_time - start_time)
        #     pass
        #     start_time = time.time()

        # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
        #     pass

        if path is not None:
            break

    pos_registry_new, actions = dict(), list()

    # now we have to extract all the tuples in the path, to convert them to movements
    # ideally, we should group them to avoid hitting the 20 instructions limit
    # todo: if destination too far, only go to a closer one and warn the called somehow? probably changes the signature

    if path is None:  # can't find decent path... let's waste time by staying static n turns?
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
                max_dumb_moves=0., step_on_resource_penalty=1.1, step_on_factory_center_penalty=1.2):

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
        return [], 0  # we made it!!!!

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

        path, cost = move_to_rec(unit, target_pos, game_state, position_registry,
                                 unit_pos=next_pos,
                                 turn_counter=turn_counter+1, dumb_moves_counter=tmp_dumb_moves_counter,
                                 max_dumb_moves=max_dumb_moves)
        if path is not None:
            rubble = game_state.board.rubble[next_x, next_y]
            extra_cost = int(np.floor(unit_cfg.MOVE_COST + rubble * unit_cfg.RUBBLE_MOVEMENT_COST))
            return [opt[0], *path], cost + extra_cost

    # once all options have been explored, can attempt the semi discarded ones (ice / ore / factory_middle)
    for opt in [*last_resort_options, *very_last_resort_options]:
        tmp_dumb_moves_counter = dumb_moves_counter + opt[1]
        next_x, next_y = unit_pos[0] + opt[0][0], unit_pos[1] + opt[0][1]
        next_pos = np.array([next_x, next_y])
        path, cost = move_to_rec(unit, target_pos, game_state, position_registry,
                                 unit_pos=next_pos,
                                 turn_counter=turn_counter+1, dumb_moves_counter=tmp_dumb_moves_counter,
                                 max_dumb_moves=max_dumb_moves)
        if path is not None:
            rubble = game_state.board.rubble[next_x, next_y]
            extra_cost = int(np.floor(unit_cfg.MOVE_COST + rubble * unit_cfg.RUBBLE_MOVEMENT_COST))
            return [opt[0], *path], cost + extra_cost

    return None, 0
