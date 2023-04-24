import collections
import numpy as np
import random
import time
import scipy
import sys

from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from lux.utils_raoul import manhattan_dist_vect_point, manhattan_dist_points, chebyshev_dist_points, \
    chebyshev_dist_vect_point, weighted_rubble_adjacent_density, score_rubble_tiles_to_dig, water_cost_to_end, \
    assess_units_main_threat, is_registry_free, find_sweet_attack_spots_on_factory, \
    find_sweet_rest_spots_nearby_factory, get_unit_next_action_as_tuple, delete_value_from_dict, \
    manh_dist_to_factory_points, manh_dist_to_factory_vect_point
from agent_tricks import assist_adj_digging, go_adj_dig, go_to_factory, go_dig_resource, go_dig_rubble, go_bully, \
    go_resist2, attack_opponent_lichen
from assignments import decide_factory_regime, assign_factories_resource_tiles, \
    update_assignments, get_ideal_assignment_queue


# debug purposes
monitored_units = []
monitored_turns = []
time_monitored_turns = []

max_distance_to_exploit_ore = 10
max_distance_to_exploit_ice = 5


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        self.faction_names = {
            'player_0': 'TheBuilders',
            'player_1': 'FirstMars'
        }

        # below: define persistent memory, that is kept from a round to the next one
        # has to be maintained to stay up to date (units/factories can be destroyed etc)
        self.robots_assignments = dict()  # u_id: assignment
        self.map_unit_factory = dict()  # u_id: f_id
        self.factory_resources_map = None  # f_id: [resources_tiles]
        self.rubble_tiles_being_dug = dict()  # u_id: tuple(tile)
        self.position_registry = dict()  # (turn, (x, y)): u_id
        self.fighting_units = list()  # list of fighting u_id  (remembered to de assign tasks if no more fighting)
        self.factory_regimes = dict()  # f_id: ["high_level_priorities"]
        self.bullies_register = dict()  # u_id: np.array(bully_tile)

    # def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
    #     """
    #     Early Phase
    #     """
    #
    #     actions = dict()
    #     if step == 0:
    #         # Declare faction
    #         actions['faction'] = self.faction_names[self.player]
    #         actions['bid'] = 0 if self.player == "player_0" else 1  # to do: try different things here
    #     else:
    #         # Factory placement period
    #         # optionally convert observations to python objects with utility functions
    #         game_state = obs_to_game_state(step, self.env_cfg, obs)
    #         op_factory_tiles = [f.pos for _, f in game_state.factories[self.opp_player].items()]
    #         my_factory_tiles = [f.pos for _, f in game_state.factories[self.player].items()]
    #         all_factory_tiles = op_factory_tiles + my_factory_tiles
    #
    #         # how much water and metal you have in your starting pool to give to new factories
    #         water_left = game_state.teams[self.player].water
    #         metal_left = game_state.teams[self.player].metal
    #
    #         # how many factories you have left to place
    #         factories_to_place = game_state.teams[self.player].factories_to_place
    #         my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
    #         if factories_to_place > 0 and my_turn_to_place:
    #             # we will spawn our factory in a random location with 100 metal n water (learnable)
    #             potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
    #
    #             ice_map = game_state.board.ice
    #             ore_map = game_state.board.ore
    #             ice_tile_locations = np.argwhere(ice_map == 1)  # numpy position of every ice tile
    #             ore_tile_locations = np.argwhere(ore_map == 1)  # numpy position of every ice tile
    #
    #             best_score = (False, -1)
    #             best_loc = potential_spawns[0]
    #
    #             for loc in potential_spawns:
    #
    #                 score_ice, ice_count = 0, 0
    #                 sorted_ice_tiles = sorted(ice_tile_locations, key=lambda t: manh_dist_to_factory_points(t, loc))
    #                 # sorted_ice_dist = sorted(manh_dist_to_factory_vect_point(ice_tile_locations, loc))
    #                 bonus_ice_adj = False
    #                 for ice_tile in sorted_ice_tiles:
    #                     multiplier = 1.
    #                     ice_dist = manh_dist_to_factory_points(ice_tile, loc)
    #                     if len(my_factory_tiles):
    #                         others_ice_dist = manh_dist_to_factory_vect_point(np.array(my_factory_tiles), ice_tile)
    #                         if np.min(others_ice_dist) <= ice_dist:
    #                             multiplier = 0.1  # will actually be considered as another factory's tile!
    #                     if len(op_factory_tiles):
    #                         others_ice_dist = manh_dist_to_factory_vect_point(np.array(op_factory_tiles), ice_tile)
    #                         if np.min(others_ice_dist) < ice_dist:
    #                             multiplier = 0.7  # will actually be considered as another factory's tile!
    #
    #                     ice_count += 1
    #                     if ice_count > 3 or ice_dist > max_distance_to_exploit_ice:
    #                         break
    #                     score = 0.2 + 0.8 * (max_distance_to_exploit_ice - ice_dist + 1) / max_distance_to_exploit_ice
    #                     if ice_dist == 1:
    #                         score_ice += 0.6 if not bonus_ice_adj else 0.2  # only give the full bonus once
    #                         bonus_ice_adj = True
    #                     score_ice += multiplier * score * (1 if ice_count < 3 else 0.2) * (1 if ice_count < 2 else 0.5)
    #
    #                 if score_ice < 0.4 and best_score[1] > 0:
    #                     continue  # bad location, let's keep moving
    #
    #                 score_ore, ore_count = 0, 0
    #                 sorted_ore_tiles = sorted(ore_tile_locations, key=lambda t: manh_dist_to_factory_points(t, loc))
    #                 bonus_ore_adj = False
    #                 for ore_tile in sorted_ore_tiles:
    #                     multiplier = 1.
    #                     ore_dist = manh_dist_to_factory_points(ore_tile, loc)
    #                     if len(my_factory_tiles):
    #                         others_ore_dist = manh_dist_to_factory_vect_point(np.array(my_factory_tiles), ore_tile)
    #                         if np.min(others_ore_dist) <= ore_dist:
    #                             multiplier = 0.1  # will actually be considered as another factory's tile!
    #                     if len(op_factory_tiles):
    #                         others_ore_dist = manh_dist_to_factory_vect_point(np.array(op_factory_tiles), ore_tile)
    #                         if np.min(others_ore_dist) < ore_dist:
    #                             multiplier = 0.7  # will actually be considered as another factory's tile!
    #
    #                     ore_count += 1
    #                     if ore_count > 3 or ore_dist > max_distance_to_exploit_ore:
    #                         break
    #                     score = 0.2 + 0.8 * (max_distance_to_exploit_ore - ore_dist + 1) / max_distance_to_exploit_ore
    #                     if ore_dist == 1:
    #                         score_ore += 0.6 if not bonus_ore_adj else 0.2  # only give the full bonus once
    #                         bonus_ore_adj = True
    #                     score_ore += multiplier * score * (1 if ore_count < 3 else 0.3) * (1 if ore_count < 2 else 0.5)
    #
    #                 density_rubble = weighted_rubble_adjacent_density(game_state, loc)
    #                 mixed_score = score_ice * 1.5 + score_ore - density_rubble/100 * 0.6
    #                 # mixed_score = 0.75 * score_ice + (score_ore + 0.2) - density_rubble / 100 * 1.0
    #                 # mixed_score = 0.75 * score_ice + (score_ore + 0.2) - density_rubble / 100 * 1.0
    #                 ore_score = score_ore - density_rubble / 100 * 0.8
    #                 tile_score = (score_ice >= 1, mixed_score) if score_ice >= 1 else (score_ice >= 1, ore_score)
    #
    #                 if tile_score > best_score:
    #                     best_score = tile_score
    #                     best_loc = loc
    #
    #             #                 spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
    #             spawn_loc = best_loc
    #             actions['spawn'] = spawn_loc
    #             #                 actions['metal']=metal_left
    #             #                 actions['water']=water_left
    #             # actions['metal'] = min(300, metal_left)
    #             # actions['water'] = min(300, water_left)
    #             # actions["metal"] = min(int(1.0 * metal_left / factories_to_place), metal_left)
    #             # actions["water"] = min(int(1.0 * water_left / factories_to_place), water_left)
    #             actions['metal'] = min(150, metal_left)
    #             actions['water'] = min(150, water_left)
    #
    #             # actions["metal"] = min(max(min(metal_left - 100 * factories_to_place, 200), 100), metal_left)
    #             # actions["water"] = min(int(1.1 * water_left / factories_to_place), water_left)
    #
    #     return actions

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Early Phase
        """

        actions = dict()
        if step == 0:
            # Declare faction
            actions['faction'] = self.faction_names[self.player]
            actions['bid'] = 10
        else:
            # Factory placement period
            # optionally convert observations to python objects with utility functions
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            opp_factories = [f.pos for _, f in game_state.factories[self.opp_player].items()]
            my_factories = [f.pos for _, f in game_state.factories[self.player].items()]

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 100 metal n water (learnable)
                potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))

                ice_map = game_state.board.ice
                ore_map = game_state.board.ore
                ice_tile_locations = np.argwhere(ice_map == 1)  # numpy position of every ice tile
                ore_tile_locations = np.argwhere(ore_map == 1)  # numpy position of every ice tile

                min_score = 10e6
                best_loc = potential_spawns[0]

                d_rubble = 10

                for loc in potential_spawns:

                    # ice_tile_distances = np.mean((ice_tile_locations - loc) ** 2, 1)
                    # ore_tile_distances = np.mean((ore_tile_locations - loc) ** 2, 1)

                    # ice_dist = sorted(set(manhattan_dist_vect_point(ice_tile_locations, loc)))
                    # ice_dist = sorted(manhattan_dist_vect_point(ice_tile_locations, loc))
                    # ice_dist = sorted(chebyshev_dist_vect_point(ice_tile_locations, loc))

                    # reflect best the need to be adjacent to a water tile when possible
                    custom_made_dist = np.floor(
                        (manhattan_dist_vect_point(ice_tile_locations, loc) +
                         chebyshev_dist_vect_point(ice_tile_locations, loc)) / 2).astype(int)
                    ice_dist = sorted(custom_made_dist)

                    nearest_ice_dist, second_nearest_ice_dist = ice_dist[0], ice_dist[1]
                    nearest_ore_dist = np.min(manhattan_dist_vect_point(ore_tile_locations, loc))

                    if 10 * nearest_ice_dist > min_score:
                        continue  # skip exploration of too crappy tiles

                    density_rubble = weighted_rubble_adjacent_density(game_state, loc)

                    closes_opp_factory_dist = 0
                    if len(opp_factories) >= 1:
                        closes_opp_factory_dist = np.min(manhattan_dist_vect_point(np.array(opp_factories), loc))
                    closes_my_factory_dist = 0
                    if len(my_factories) >= 1:
                        closes_my_factory_dist = np.min(manhattan_dist_vect_point(np.array(opp_factories), loc))

                    score = 10 * nearest_ice_dist + 0.2 * second_nearest_ice_dist +\
                            1. * nearest_ore_dist + \
                            0.3 * density_rubble - \
                            0.1 * closes_opp_factory_dist + 0.1 * closes_my_factory_dist

                    if score < min_score:
                        min_score = score
                        best_loc = loc

                #                 spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
                spawn_loc = best_loc
                actions['spawn'] = spawn_loc
                #                 actions['metal']=metal_left
                #                 actions['water']=water_left
                # actions['metal'] = min(300, metal_left)
                # actions['water'] = min(300, water_left)
                actions["metal"] = min(int(1.0 * metal_left / factories_to_place), metal_left)
                actions["water"] = min(int(1.0 * water_left / factories_to_place), water_left)

                # actions["metal"] = min(max(min(metal_left - 100 * factories_to_place, 200), 100), metal_left)
                # actions["water"] = min(int(1.1 * water_left / factories_to_place), water_left)

        return actions

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()

        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories, units = game_state.factories[self.player], game_state.units[self.player]
        factory_tiles = np.array([f.pos for f_id, f in factories.items()])
        all_factories = [f for f_id, f in factories.items()]
        factories_power = {f_id: f.power for f_id, f in factories.items()}
        op_player = "player_1" if self.player == "player_0" else "player_0"
        # if game_state.real_env_steps in monitored_turns:
        #     pass
        main_threats = assess_units_main_threat(units, game_state.units[op_player])
        cur_turn = game_state.real_env_steps

        #################################################################################
        #               ASSIGNMENT HANDLING
        #################################################################################
        # update persistent state in case of death of units or factories
        self.robots_assignments = {u_id: asgnmt for u_id, asgnmt in self.robots_assignments.items()
                                   if u_id in units.keys()}
        self.bullies_register = {u_id: t for u_id, t in self.bullies_register.items() if u_id in units.keys()}
        self.map_unit_factory = {u_id: f_id for u_id, f_id in self.map_unit_factory.items()
                                 if u_id in units.keys() and f_id in factories.keys()}

        # identify fighting units which are safe now...
        not_fighting_anymore = [u_id for u_id in self.fighting_units if u_id in main_threats.keys() and
                                (main_threats[u_id] is None or main_threats[u_id][1] > 2)]
        for u_id in not_fighting_anymore:
            actions[u_id] = list()  # free to do whatever now (registry cleaned just below)
        self.fighting_units = list()  # will be reassessed dynamically below...
        self.position_registry = {(turn, (x, y)): u_id for (turn, (x, y)), u_id in self.position_registry.items()
                                  if u_id in units.keys() and turn >= cur_turn and u_id not in not_fighting_anymore}

        if game_state.real_env_steps in time_monitored_turns:
            start_time = time.time()

        # todo: rewrite with: for k in list(keys()) { (...) del d[k] } will reduce cost to n from n^2
        for (turn, (x, y)), unit_id in self.position_registry.items():
            if turn == cur_turn and tuple(units[unit_id].pos) != (x, y):
                # print("lost unit: ", unit_id, "    on turn:", game_state.real_env_steps)
                # reset the unit, as something happened leading to the unit not to be where it should (power issue?)
                self.position_registry = {key: u_id for key, u_id in self.position_registry.items() if u_id != unit_id}
                actions[unit_id] = list()

        # associate all resources with a unique factory (refreshed only if not done or if a factory disappears)
        if self.factory_resources_map is None or \
                any([f_id not in factories.keys() for f_id in self.factory_resources_map]):
            self.factory_resources_map = assign_factories_resource_tiles(factories, game_state)

        # ensure all units have an assigned factory
        for unit_id, unit in units.items():
            if unit_id in self.robots_assignments.keys() and unit_id in self.map_unit_factory.keys() and \
                    self.map_unit_factory[unit_id] in factories.keys():
                continue  # nothing to do, all in order
            factory_distances = manhattan_dist_vect_point(factory_tiles, unit.pos)
            near_factory_i = np.argmin(factory_distances)  # todo: crash if no factories. not a big deal? (lost anyway)
            if unit_id not in self.map_unit_factory.keys() or self.map_unit_factory[unit_id] not in factories.keys():
                self.map_unit_factory[unit_id] = all_factories[near_factory_i].unit_id  # assign factory to unit
                if unit_id in self.robots_assignments.keys():
                    del self.robots_assignments[unit_id]  # new factory associated, let's remove assignment

        # review and reorganise assignments if necessary (includes death of factories)
        assignment_tile_map_per_factory, ideal_queue_per_factory = dict(), dict()
        for factory_id, factory in factories.items():
            # factory_units_ids = [u_id for u_id, f_id in self.map_unit_factory.items() if f_id == factory_id]
            factory_units = {u_id: u for u_id, u in units.items()
                             if u_id in self.map_unit_factory.keys() and self.map_unit_factory[u_id] == factory_id}
            fact_units_assignments = {u_id: asgnmt for u_id, asgnmt in self.robots_assignments.items()
                                      if u_id in factory_units.keys()}
            if (cur_turn % 25) == 0 or not len(self.factory_regimes):
                self.factory_regimes[factory_id] = decide_factory_regime(
                    factory, game_state, factory_units, self.factory_resources_map,
                    max_distance_to_exploit_ore=max_distance_to_exploit_ore,
                    max_distance_to_exploit_ice=max_distance_to_exploit_ice)
            ideal_queue_per_factory[factory_id], assignment_tile_map_per_factory[factory_id] = \
                get_ideal_assignment_queue(factory, game_state, self.factory_resources_map,
                                           self.factory_regimes[factory_id])
            # if cur_turn == 124 and factory_id == "factory_3":
            #     pass
            updated_assignments_d = update_assignments(factory, factory_units, fact_units_assignments,
                                                       ideal_queue_per_factory[factory_id], heavy_reassign=True)
            # if cur_turn == 124 and factory_id == "factory_3":
            #     pass
            # delete future actions and positions so it's recomputed
            for (unit_id, updated_asgnmt) in updated_assignments_d.items():
                # if unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
                #     pass
                actions[unit_id] = list()

            self.position_registry = {key: u_id for key, u_id in self.position_registry.items()
                                      if u_id not in updated_assignments_d.keys()}
            self.robots_assignments.update(updated_assignments_d)

        # once all assignments have been done / redone, update rubble cache
        # delete entry if unit died or if not digging anymore
        self.rubble_tiles_being_dug = {u_id: t for u_id, t in self.rubble_tiles_being_dug.items() if
                                       u_id in units.keys() and self.robots_assignments[u_id].startswith("dig_rubble")
                                       and len(units[u_id].action_queue)}

        # define bullying tiles of interest
        resources_bully_tiles, lichen_bully_tiles = list(), list()
        op_rest_tiles = list()
        for op_f_id, op_factory in game_state.factories[op_player].items():
            res_tiles, lich_tiles = find_sweet_attack_spots_on_factory(op_factory, game_state)
            resources_bully_tiles.extend(res_tiles)
            lichen_bully_tiles.extend(lich_tiles)
            op_rest_tiles.extend(find_sweet_rest_spots_nearby_factory(op_factory, game_state))
        my_rest_tiles = list()
        for f_id, factory in game_state.factories[self.player].items():
            my_rest_tiles.extend(find_sweet_rest_spots_nearby_factory(factory, game_state))

        self.bullies_register = {u_id: tile for u_id, tile in self.bullies_register.items()
                                 if self.robots_assignments[u_id].startswith("bully_")}

        # maintain bully register in case of opponent factory destroyed
        for unit_id in list(self.bullies_register.keys()):
            bullied_tile = self.bullies_register[unit_id]
            if bullied_tile not in resources_bully_tiles + lichen_bully_tiles + op_rest_tiles + my_rest_tiles:
                actions[unit_id] = list()  # reset bully
                del self.bullies_register[unit_id]
                self.position_registry = {key: u_id for key, u_id in self.position_registry.items() if u_id != unit_id}

        if game_state.real_env_steps in time_monitored_turns:
            end_time = time.time()
            local_time = str(end_time-start_time)
            pass
            start_time = time.time()

        #################################################################################
        #               FACTORIES CREATE UNITS ?
        #################################################################################
        for f_id, factory in factories.items():
            if is_registry_free(cur_turn + 1, factory.pos, self.position_registry, factory.unit_id):
                if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                        factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                    # if cur_turn == 0:
                    actions[f_id] = factory.build_heavy()
                    self.position_registry[(cur_turn + 1, tuple(factory.pos))] = factory.unit_id

                # unit creation
                elif factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
                        factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST:

                    # build a light unit only if there is not a unit currently digging ore (else we prefer HEAVY)
                    f_units = {u_id: u for u_id, u in units.items() if u_id in self.map_unit_factory.keys()
                               and self.map_unit_factory[u_id] == f_id}
                    f_units_assignments = {u_id: asgnmt for u_id, asgnmt in self.robots_assignments.items()
                                           if u_id in f_units.keys()}
                    next_asgnmt_needed = None
                    for asgnmt in ideal_queue_per_factory[f_id]:
                        if asgnmt not in f_units_assignments.values():
                            next_asgnmt_needed = asgnmt
                            break

                    # if cur_turn in list(range(450, 1000)):
                    #     pass
                    nb_heavy = sum([u.unit_type == "HEAVY" for u in f_units.values()])
                    # ore_to_be_dug = any(["ore" in asgnmt for asgnmt in ideal_queue_per_factory[f_id]])
                    # is_ore_assigned = any(["ore_" in asgnmt for asgnmt in f_units_assignments.values()])

                    # n_rubble_assist = sum(["assist_" in asgnmt or "dig_rubble" in asgnmt for asgnmt in
                    #                        f_units_assignments.values()])
                    # if cur_turn in (1, 2, 3, 4, 5) or (len(f_units) < 9) or \
                    #         (next_asgnmt_needed is not None and ("assist_" in next_asgnmt_needed or
                    #                                              "dig_rubble" in next_asgnmt_needed)):
                    # if cur_turn in (1, 2, 3, 4, 5) or (len(f_units) < 9) or \
                    #         (next_asgnmt_needed is not None and ("bully_" not in next_asgnmt_needed)):
                    if cur_turn in (1, 2, 3, 4, 5) or (len(f_units) - nb_heavy < 6):
                        actions[f_id] = factory.build_light()
                        self.position_registry[(cur_turn + 1, tuple(factory.pos))] = factory.unit_id

        #################################################################################
        #               UNITS ACTIONS HANDLING
        #################################################################################
        # we go by "priority" because of access to power, "important" units first
        unit_ids_by_priority = sorted(units.keys(), key=lambda u_id: ideal_queue_per_factory[
            self.map_unit_factory[u_id]].index(self.robots_assignments[u_id]))

        # fo all the other units moves
        reset_unit_ids = list()
        for unit_id in unit_ids_by_priority:

            if game_state.real_env_steps in time_monitored_turns:
                start_time = time.time()

            unit = units[unit_id]
            unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
            assignment = self.robots_assignments[unit_id]
            assigned_factory = factories[self.map_unit_factory[unit_id]]

            if unit_id in reset_unit_ids:
                # the unit has been reset by someone else to steal a tile... have to find something to do again...
                actions[unit_id] = list()

            is_unit_charging = len(unit.action_queue) and tuple(unit.action_queue[0]) == tuple(
                unit.recharge(unit_cfg.BATTERY_CAPACITY))
            if is_unit_charging:
                # we pre-booked the cell for n more turns, we free it now (will be rebooked instantly if still relevant)
                delete_value_from_dict(self.position_registry, unit_id)

            if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
                pass

            is_unit_reset = unit_id in actions.keys() and len(actions[unit_id]) == 0
            is_unit_busy = len(unit.action_queue) and (not is_unit_reset) and (not is_unit_charging)

            if is_unit_busy:  # if already have something to do, and it has not been overwritten/nullified before
                # running low on power, abort and go back to factory...
                # should be a proper function to check for a create this interruption event

                if "solo" in assignment and unit.power / unit_cfg.DIG_COST < 2 and \
                        any([act[0] == 3 for act in unit.action_queue]):  # only interrupt if digging (act[0] == 3)
                    self.position_registry = {k: u_id for k, u_id in self.position_registry.items()
                                              if u_id != unit_id}  # remove plan, make another one
                    actions[unit_id] = go_to_factory(unit, game_state, self.position_registry, assigned_factory)
                # if unit.unit_id == "unit_15" and game_state.real_env_steps >= 133:
                #     pass
                if "duo_main" in assignment and unit.power / unit_cfg.DIG_COST < 2 and \
                        any([act[0] == 3 for act in unit.action_queue]):   # only interrupt if digging (act[0] == 3)
                    resource, n_allocated = assignment.split('_')[0], int(assignment.split('_')[-1])
                    u_id_assist_l = [u_id for u_id, f_id in self.map_unit_factory.items() if
                                     f_id == self.map_unit_factory[unit.unit_id] and
                                     self.robots_assignments[u_id] == f"{resource}_duo_assist_{n_allocated}"]
                    assistant_unit = units[u_id_assist_l[0]] if len(u_id_assist_l) else None
                    if assistant_unit is None:
                        self.position_registry = {k: u_id for k, u_id in self.position_registry.items()
                                                  if u_id != unit_id}  # remove plan, make another one
                        actions[unit_id] = go_to_factory(unit, game_state, self.position_registry, assigned_factory)
                if game_state.real_env_steps in time_monitored_turns:
                    end_time = time.time()
                    local_time = str(round(float(end_time - start_time), 3))
                    if (end_time - start_time) > 1.:
                        with open('data/a_monitor_execution_time_by_unit.txt', 'a') as f:
                            f.write(str(game_state.real_env_steps) + "[" + str(unit_id) + "]: " + local_time + "\n")
                        pass
                continue  # stop here if already have some actions to do

            # so unit is not busy, let's do something
            elif unit_id not in actions.keys():
                actions[unit_id] = list()

            if assignment.startswith("water_duo_main") or assignment.startswith("ore_duo_main"):
                dest_tile = assignment_tile_map_per_factory[self.map_unit_factory[unit_id]][assignment]
                actions[unit_id].extend(go_adj_dig(unit, game_state, self.position_registry, dest_tile,
                                                   assigned_factory=assigned_factory, factories_power=factories_power))
            elif assignment.startswith("water_duo_assist"):
                n_allocated = int(assignment.split('_')[-1])
                u_id_main = [u_id for u_id, f_id in self.map_unit_factory.items() if
                             f_id == self.map_unit_factory[unit.unit_id] and
                             self.robots_assignments[u_id] == f"water_duo_main_{n_allocated}"][0]
                assisted_unit = units[u_id_main]

                dug_tile = assignment_tile_map_per_factory[self.map_unit_factory[unit_id]][
                    self.robots_assignments[assisted_unit.unit_id]]
                actions[unit_id].extend(assist_adj_digging(
                    unit, game_state, self.position_registry, assisted_unit, assigned_factory=assigned_factory,
                    factories_power=factories_power, target_dug_tile=dug_tile))
            elif assignment.startswith("ore_duo_assist"):
                n_allocated = int(assignment.split('_')[-1])
                u_id_main = [u_id for u_id, f_id in self.map_unit_factory.items() if
                             f_id == self.map_unit_factory[unit.unit_id] and
                             self.robots_assignments[u_id] == f"ore_duo_main_{n_allocated}"][0]
                assisted_unit = units[u_id_main]
                dug_tile = assignment_tile_map_per_factory[self.map_unit_factory[unit_id]][
                    self.robots_assignments[assisted_unit.unit_id]]
                actions[unit_id].extend(assist_adj_digging(
                    unit, game_state, self.position_registry, assisted_unit, assigned_factory=assigned_factory,
                    factories_power=factories_power, target_dug_tile=dug_tile))
            elif assignment.startswith("ore_solo") or assignment.startswith("water_solo"):
                dug_tile = assignment_tile_map_per_factory[self.map_unit_factory[unit_id]][assignment]
                actions[unit_id].extend(go_dig_resource(
                    unit, game_state, self.position_registry, dug_tile, assigned_factory, factories_power,
                    unit_ids_by_priority, reset_unit_ids))
            elif assignment.startswith("dig_rubble"):
                # tiles_scores = score_rubble_tiles_to_dig(game_state, assigned_factory)
                actions[unit_id].extend(go_dig_rubble(
                    unit, game_state, self.position_registry, assigned_factory, factories_power,
                    self.rubble_tiles_being_dug, tiles_scores=None, n_min_digs=5, n_desired_digs=8))
            elif assignment.startswith("bully"):
                actions[unit_id].extend(go_bully(
                    unit, game_state, self.position_registry, assigned_factory, self.bullies_register, factories_power,
                    my_rest_tiles, op_rest_tiles, resources_bully_tiles, lichen_bully_tiles))
            else:
                # if np.array_equal(unit.pos, assigned_factory.pos):
                #     actions[unit_id].extend([unit.move(1)])
                # actions[unit_id].extend([unit.recharge(x=3 * unit_cfg.DIG_COST)])
                # debug purposes only!  # todo: replace that self destruct
                actions[unit_id].extend([unit.self_destruct(repeat=0, n=1)])

            if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
                pass

            if game_state.real_env_steps in time_monitored_turns:
                end_time = time.time()
                local_time = str(round(float(end_time-start_time), 3))
                if (end_time-start_time) > 1.:
                    with open('data/a_monitor_execution_time_by_unit.txt', 'a') as f:
                        f.write(str(game_state.real_env_steps) + "[" + str(unit_id) + "]: " + local_time + "\n")
                    pass

        if game_state.real_env_steps in time_monitored_turns:
            end_time = time.time()
            local_time = str(end_time - start_time)
            pass

        ################################################################################
        #              END GAME AGGRESSIVITY
        ################################################################################
        for unit_id in unit_ids_by_priority:
            unit = units[unit_id]
            assignment = self.robots_assignments[unit_id]
            assigned_factory = factories[self.map_unit_factory[unit_id]]
            unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
            should_fight = unit.power >= (1000 - game_state.real_env_steps) * (
                    unit_cfg.DIG_COST + unit_cfg.MOVE_COST) / 2.6
            if should_fight:
            # if not assignment.startswith("bully_") and should_fight:
                attack_lichen_options = attack_opponent_lichen(
                    unit, game_state, self.position_registry, assigned_factory, control_area=4)
                if attack_lichen_options is not None:
                    # very ugly code, because we've been lazy with hypothetical moves in agent_tricks
                    # we need to delete the registry if we actually want to fight and cancel previous plan
                    delete_value_from_dict(self.position_registry, unit_id)
                    attack_lichen_options = attack_opponent_lichen(
                        unit, game_state, self.position_registry, assigned_factory, control_area=4)
                    actions[unit_id] = attack_lichen_options

        #################################################################################
        #               UNITS DEFENSE
        #################################################################################
        for unit_id in unit_ids_by_priority[::-1]:
            if main_threats[unit_id] is not None and main_threats[unit_id][1] <= 2:
                unit = units[unit_id]
                next_unit_action = get_unit_next_action_as_tuple(unit, actions)
                # unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
                # assignment = self.robots_assignments[unit_id]
                assigned_factory = factories[self.map_unit_factory[unit_id]]
                threat_unit = game_state.units[op_player][main_threats[unit_id][0]]
                dist_to_threat = main_threats[unit_id][1]
                resist_actions = go_resist2(unit, game_state, self.position_registry, threat_unit,
                                            assigned_factory, dist_to_threat, next_unit_action, factories_power)
                if resist_actions is not None:
                    actions[unit_id] = resist_actions
                    self.fighting_units.append(unit_id)

        #################################################################################
        #               LAST TURNS 'CHEESE'
        #################################################################################
        for unit_id in unit_ids_by_priority:
            unit = units[unit_id]
            unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
            lichen_on_tile = game_state.board.lichen[unit.pos[0], unit.pos[1]]
            op_strain_ids = [f.strain_id for f in game_state.factories[op_player].values()]
            if cur_turn in [998, 999] and lichen_on_tile > 0 and \
                    game_state.board.lichen_strains[unit.pos[0], unit.pos[1]] in op_strain_ids:
                if unit.power > unit_cfg.SELF_DESTRUCT_COST + unit_cfg.ACTION_QUEUE_POWER_COST:
                    if unit.unit_type == "HEAVY" and cur_turn == 998:
                        continue  # edge case, unit could be useful twice...
                    actions[unit_id] = [unit.self_destruct()]  # BOOM!
                elif unit.power > unit_cfg.DIG_COST + unit_cfg.ACTION_QUEUE_POWER_COST:
                    actions[unit_id] = [unit.dig(1)]

        #################################################################################
        #               FACTORY WATERING
        #################################################################################
        for f_id, factory in factories.items():
            if f_id in actions.keys() or not factory.can_water(game_state):
                continue  # we already make a unit, can't water this turn

            current_water_cost = max(factory.water_cost(game_state), 1)
            projected_total_cost = water_cost_to_end(current_water_cost, 1000 - cur_turn, increase_rate=0.015) + 5
            if factory.cargo.water > projected_total_cost and cur_turn > 300:
                actions[f_id] = factory.water()
            elif factory.cargo.water > 200 and ((step % 2) == 0 or (step % 11) == 0):
                actions[f_id] = factory.water()  # attempt to get a bit more power for cheap
            else:
                f_units = {u_id: u for u_id, u in units.items() if u_id in self.map_unit_factory.keys()
                           and self.map_unit_factory[u_id] == f_id}
                f_units_assignments = {u_id: asgnmt for u_id, asgnmt in self.robots_assignments.items()
                                       if u_id in f_units.keys()}
                is_water_heavy_dug = any([u.unit_type == "HEAVY" and "water_" in f_units_assignments[u_id] and
                                          "assist_" not in f_units_assignments[u_id] for u_id, u in f_units.items()])
                # if is_water_heavy_dug and factory.cargo.water > 160 and cur_turn < 250 and factory.power < 200:
                #     actions[f_id] = factory.water()
                if factory.cargo.water > 350 and factory.power < 250:# and cur_turn < 300:
                    actions[f_id] = factory.water()
                    # unsure about that one... should at least check if the water is currently dug, i.e. unit on tile

        # cheap action optimiser!
        for u_id in list(actions.keys()):
            if u_id in factories.keys():
                continue  # (special because not iterable...)
            if len(actions[u_id]) > 20:
                actions[u_id] = actions[u_id][:20]
            elif len(actions[u_id]) < 19:  # small optim; attempt to plan next round
                if actions[u_id][-1][0] != 5 and all([act[-2] == 0 for act in actions[u_id]]):
                    # not ending by charge, no repeat in the loop: we add a charge at the end (i.e. most likely outcome)
                    unit, unit_cfg = units[u_id], game_state.env_cfg.ROBOTS[units[u_id].unit_type]
                    actions[u_id].extend([unit.recharge(unit_cfg.BATTERY_CAPACITY)])
            if tuple(tuple(a) for a in actions[u_id]) == tuple(tuple(a) for a in units[u_id].action_queue):
                del actions[u_id]

        if cur_turn in monitored_turns:
            pass

        return actions
