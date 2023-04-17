import collections
import numpy as np
import time
import scipy
import sys

from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from lux.utils_raoul import manhattan_dist_vect_point, manhattan_dist_points, chebyshev_dist_points, \
    chebyshev_dist_vect_point, weighted_rubble_adjacent_density, score_rubble_tiles_to_dig, water_cost_to_end, \
    assess_units_main_threat, is_unit_stronger, is_unit_safe, is_registry_free
from agent_tricks import assist_adj_digging, go_adj_dig, go_to_factory, go_dig_resource, go_dig_rubble, go_fight
from assignments import update_assignments_old, give_assignment, get_ideal_assignment_queue, \
    assign_factories_resource_tiles, update_assignments_new, get_ideal_assignment_queue_new


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
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

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        """
        Early Phase
        """

        actions = dict()
        if step == 0:
            # Declare faction
            actions['faction'] = self.faction_names[self.player]
            actions['bid'] = 0  # Learnable
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

                    density_rubble = weighted_rubble_adjacent_density(obs["board"]["rubble"], loc)

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

        factories_power = dict()  # round state variable to keep track of power expenses

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        units = game_state.units[self.player]
        # game_state.teams[self.player].place_first
        factory_tiles, factory_units = [], []
        op_player = "player_1" if self.player == "player_0" else "player_0"
        main_threats = assess_units_main_threat(units, game_state.units[op_player])

        overall_time_monitored_turns = []
        if game_state.env_steps in overall_time_monitored_turns:
            start_time = time.time()

        for unit_id, factory in factories.items():

            # prepare variables
            factory_tiles += [factory.pos]
            factory_units += [factory]
            factories_power[unit_id] = factory.power

            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                    factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST and \
                    (game_state.real_env_steps == 0):
                actions[unit_id] = factory.build_heavy()

            # todo: unit creation that makes sense
            if factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
                    factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST and \
                    (game_state.real_env_steps in (1, 2, 3, 4, 5)):
                actions[unit_id] = factory.build_light()

            current_water_cost = max(factory.water_cost(game_state), 1)
            projected_total_cost = water_cost_to_end(current_water_cost, 1000 - game_state.real_env_steps,
                                                     increase_rate=0.015) + 10
            if factory.can_water(game_state) and factory.cargo.water > projected_total_cost and \
                    game_state.real_env_steps > 500:
                actions[unit_id] = factory.water()
            elif factory.can_water(game_state) and factory.cargo.water > 160 and (
                    (step % 2) == 0 or (step % 11) == 0):
                actions[unit_id] = factory.water()  # attempt to get a bit more power for cheap
        factory_tiles = np.array(factory_tiles)

        #################################################################################
        #               ASSIGNMENT HANDLING
        #################################################################################
        # time_monitored_turns = [162, 164]
        # time_monitored_turns = [875, ]
        time_monitored_turns = []

        # update persistent state in case of death of units or factories
        self.robots_assignments = {u_id: asgnmt for u_id, asgnmt in self.robots_assignments.items()
                                   if u_id in units.keys()}
        self.map_unit_factory = {u_id: f_id for u_id, f_id in self.map_unit_factory.items()
                                 if u_id in units.keys() and f_id in factories.keys()}

        not_fighting_anymore = [u_id for u_id in self.fighting_units if u_id in main_threats.keys() and
                                main_threats[u_id][1] > 2]  # identify fighting units which are safe now...
        for u_id in not_fighting_anymore:
            actions[u_id] = list()  # free to do whatever now (registry cleaned just below)
        self.fighting_units = list()  # will be reassessed dynamically below...
        self.position_registry = {(turn, (x, y)): u_id for (turn, (x, y)), u_id in self.position_registry.items()
                                  if u_id in units.keys() and turn >= game_state.real_env_steps
                                  and u_id not in not_fighting_anymore}

        if game_state.env_steps in time_monitored_turns:
            start_time = time.time()

        for (turn, (x, y)), unit_id in self.position_registry.items():
            if turn == game_state.real_env_steps and tuple(units[unit_id].pos) != (x, y):
                # print("lost unit: ", unit_id, "    on turn:", game_state.real_env_steps)
                # reset the unit, as something happened leading to the unit not to be where it should (power issue?)
                self.position_registry = {key: u_id for key, u_id in self.position_registry.items() if u_id != unit_id}
                actions[unit_id] = list()

        if game_state.env_steps in time_monitored_turns:
            end_time = time.time()
            local_time = str(end_time-start_time)
            pass
            start_time = time.time()

        # initialise dictionary that keeps track of good tiles to dig around factories
        # rubble_tiles_scores_per_factory = dict()  # todo: could be used as a cache for a given round to save time

        # associate all resources with a unique factory (refreshed only if not done or if a factory disappears)
        if self.factory_resources_map is None or \
                any([f_id not in factories.keys() for f_id in self.factory_resources_map]):
            self.factory_resources_map = assign_factories_resource_tiles(factories, game_state)

        # ensure all units have a factory and an assignment
        for unit_id, unit in units.items():

            if unit_id in self.robots_assignments.keys() and unit_id in self.map_unit_factory.keys() and \
                    self.map_unit_factory[unit_id] in factories.keys():
                continue  # nothing to do, all in order

            factory_distances = manhattan_dist_vect_point(factory_tiles, unit.pos)
            near_factory_i = np.argmin(factory_distances)  # todo: crash if no factories. not a big deal? (lost anyway)

            if unit_id not in self.map_unit_factory.keys() or self.map_unit_factory[unit_id] not in factories.keys():
                self.map_unit_factory[unit_id] = factory_units[near_factory_i].unit_id  # assign factory to unit
                if unit_id in self.robots_assignments.keys():
                    del self.robots_assignments[unit_id]  # new factory associated, let's remove assignment

            # if unit_id not in self.robots_assignments.keys():
            #     assigned_factory = factories[self.map_unit_factory[unit_id]]
                # ideal_queue, _ = get_ideal_assignment_queue(assigned_factory, game_state,
                #                                             assigned_resources=self.factory_resources_map)
                # assignment = give_assignment(unit, game_state, self.robots_assignments, self.map_unit_factory,
                #                              assigned_resources=self.factory_resources_map,
                #                              ideal_queue=ideal_queue, assigned_factory=factory_units[near_factory_i])
                # self.robots_assignments[unit_id] = assignment
                # self.map_unit_factory[unit_id] = factory_units[near_factory_i].unit_id

        if game_state.env_steps in time_monitored_turns:
            end_time = time.time()
            local_time = str(end_time-start_time)
            pass
            start_time = time.time()

        # these tiles will be avoided by solo resource diggers and rubble diggers
        assistant_tiles = list()  # list of tiles pre-booked for assistants

        # review and reorganise assignments if necessary (includes death of factories)
        assignment_tile_map_per_factory, ideal_queue_per_factory = dict(), dict()
        for factory_id, factory in factories.items():
            # factory_units_ids = [u_id for u_id, f_id in self.map_unit_factory.items() if f_id == factory_id]
            factory_units = {u_id: u for u_id, u in units.items()
                             if u_id in self.map_unit_factory.keys() and self.map_unit_factory[u_id] == factory_id}
            fact_units_assignments = {u_id: asgnmt for u_id, asgnmt in self.robots_assignments.items()
                                   if u_id in factory_units.keys()}
            # get_ideal_assignment_queue(agent, factory, game_state, assigned_resources
            ideal_queue_per_factory[factory_id], assignment_tile_map_per_factory[factory_id] = \
                get_ideal_assignment_queue_new(factory, game_state, self.factory_resources_map)

            updated_assignments_d = update_assignments_new(factory, factory_units, fact_units_assignments,
                                                           ideal_queue_per_factory[factory_id], heavy_reassign=True, game_state=game_state)
            # updated_assignments_d = update_assignments_old(fact_units_assignments, ideal_queue_per_factory[factory_id],
            #                                                units, heavy_reassign=True)
            # # TODO: think if cache / actions update necessary here...
            # delete future actions and positions so it's recomputed
            for (unit_id, updated_asgnmt) in updated_assignments_d.items():
                actions[unit_id] = list()
            self.position_registry = {key: u_id for key, u_id in self.position_registry.items()
                                      if u_id not in updated_assignments_d.keys()}
            self.robots_assignments.update(updated_assignments_d)
            # todo: remove all this concept of assistant tiles... (handled through position registry now)
            # assistant_tiles.extend([tile for asgnmt, tile in assignment_tile_map_per_factory[factory_id].items()
            #                         if "assist" in asgnmt])

        if game_state.env_steps in time_monitored_turns:
            end_time = time.time()
            local_time = str(end_time-start_time)
            pass
            start_time = time.time()

        # once all assignments have been done / redone, update rubble cache
        # delete entry if unit died or if not digging anymore
        self.rubble_tiles_being_dug = {u_id: t for u_id, t in self.rubble_tiles_being_dug.items() if
                                       u_id in units.keys() and self.robots_assignments[u_id].startswith("dig_rubble")
                                       and len(units[u_id].action_queue)}

        if game_state.env_steps in time_monitored_turns:
            end_time = time.time()
            local_time = str(end_time-start_time)
            pass
            start_time = time.time()

        #################################################################################
        #               UNITS ACTIONS HANDLING
        #################################################################################
        #
        # monitored_units = ["unit_13", "unit_17"]
        # monitored_turns = [156, 158, 159, 160, 164, 171]
        # monitored_units = ["unit_57"]
        # monitored_turns = [6]
        monitored_units = []
        monitored_turns = []

        # we go by "priority" because of access to power, "important" units first
        unit_ids_by_priority = sorted(units.keys(), key=lambda u_id: ideal_queue_per_factory[
            self.map_unit_factory[u_id]].index(self.robots_assignments[u_id]))

        # self-preservation first  (need to empty registry if we are going to run away...)
        # self.fighting_units = list()  # reset fighting units  # todo: reset optimisation
        for unit_id in unit_ids_by_priority:
            unit = units[unit_id]
            unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
            assigned_factory = factories[self.map_unit_factory[unit_id]]
            assignment = self.robots_assignments[unit_id]
            dist_to_factory = chebyshev_dist_points(assigned_factory.pos, unit.pos)
            if main_threats[unit_id] is not None:
                dist_to_threat = main_threats[unit_id][1]
                is_stronger = is_unit_stronger(unit, game_state.units[op_player][main_threats[unit_id][0]])
                if dist_to_threat <= 2 and (not is_stronger) and "_assist_" not in assignment:
                    if dist_to_factory > 1:
                        # ALLO HOUSTON, we have a problem,  # RUN FOR YOU LIFE, LITTLE ROBOT
                        self.position_registry = {k: u_id for k, u_id in self.position_registry.items()
                                                  if u_id != unit_id}  # remove plan, make another one
                        actions[unit_id] = go_to_factory(unit, game_state, self.position_registry, assigned_factory,
                                                         tiles_for_assistants_only=assistant_tiles)
                    else:
                        is_spot_free = is_registry_free(game_state.real_env_steps + 1, unit.pos, self.position_registry,
                                                        unit.unit_id)
                        if (is_spot_free and ((len(unit.action_queue) and (unit.action_queue[0][0] != 5)) or
                                              (not len(unit.action_queue)))):
                            # if on factory tile and not already charging, charge
                            actions[unit_id] = [unit.recharge(unit_cfg.BATTERY_CAPACITY)]
                        else:
                            pass  # wtf do we do if someone else booked that place? :(
                            # can we do NOTHING ? does it crash? # todo: test somehow
                    self.fighting_units.append(unit_id)
                    # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
                    #     pass

        # kick some butts next  (take precedence over other units moves)
        # self.fighting_units = list()  # reset fighting units  # todo: reset optimisation
        for unit_id in unit_ids_by_priority:
            unit = units[unit_id]
            assigned_factory = factories[self.map_unit_factory[unit_id]]
            assignment = self.robots_assignments[unit_id]
            if main_threats[unit_id] is not None:
                dist_to_threat = main_threats[unit_id][1]
                op_unit = game_state.units[op_player][main_threats[unit_id][0]]
                is_stronger, is_safe = is_unit_stronger(unit, op_unit), is_unit_safe(unit, op_unit)
                if dist_to_threat <= 2 and is_stronger and not is_safe and "_assist_" not in assignment:  # KILL!
                    # self.fighting_units.append(unit_id)  # need to know we're fighting to stop weird moves after?
                    self.position_registry = {k: u_id for k, u_id in self.position_registry.items()
                                              if u_id != unit_id}  # remove plan, make another one
                    actions[unit_id] = go_fight(unit, game_state, self.position_registry, op_unit, assigned_factory,
                                                main_threats[unit_id])
                    self.fighting_units.append(unit_id)
                    # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
                    #     pass
                # else, if weaker but have a chance if opponent does not move, we also go for it
                elif dist_to_threat == 1 and unit.unit_type == op_unit.unit_type and \
                        not is_safe and "_assist_" not in assignment:  # KILL!
                    # self.fighting_units.append(unit_id)  # need to know we're fighting to stop weird moves after?
                    self.position_registry = {k: u_id for k, u_id in self.position_registry.items()
                                              if u_id != unit_id}  # remove plan, make another one
                    actions[unit_id] = go_fight(unit, game_state, self.position_registry, op_unit, assigned_factory,
                                                main_threats[unit_id], allow_weaker=True)
                    self.fighting_units.append(unit_id)
                    # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
                    #     pass

        # fo all the other units moves
        for unit_id in unit_ids_by_priority:

            if unit_id in self.fighting_units:
                continue  # already handled above!

            unit = units[unit_id]
            # for unit_id, unit in units.items():
            unit_cfg = game_state.env_cfg.ROBOTS[unit.unit_type]
            assignment = self.robots_assignments[unit_id]
            assigned_factory = factories[self.map_unit_factory[unit_id]]

            if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
                pass

            if game_state.env_steps in time_monitored_turns:
                start_time = time.time()

            # if already have something to do, and it has not been overwritten/nullified before
            if len(unit.action_queue) and not (unit_id in actions.keys() and len(actions[unit_id]) == 0):

                # running low on power, abort and go back to factory...

                # should be a proper function to check for a create this interruption event
                if "solo" in assignment and unit.power / unit_cfg.DIG_COST < 2 and \
                        any([act[0] == 3 for act in unit.action_queue]):  # only interrupt if digging (act[0] == 3)
                    self.position_registry = {k: u_id for k, u_id in self.position_registry.items()
                                              if u_id != unit_id}  # remove plan, make another one
                    actions[unit_id] = go_to_factory(unit, game_state, self.position_registry, assigned_factory,
                                                     tiles_for_assistants_only=assistant_tiles)
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
                        actions[unit_id] = go_to_factory(unit, game_state, self.position_registry, assigned_factory,
                                                         tiles_for_assistants_only=assistant_tiles)
                continue  # stop here if already have some actions to do
            elif unit_id not in actions.keys():
                actions[unit_id] = list()
            # if game_state.real_env_steps >= 20 and unit.unit_id == "unit_53":
            #     pass
            # take power first if you can ?

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
                    unit, game_state, self.position_registry, dug_tile, assigned_factory=assigned_factory,
                    factories_power=factories_power, tiles_for_assistants_only=assistant_tiles))
            elif assignment.startswith("dig_rubble"):
                # tiles_scores = score_rubble_tiles_to_dig(game_state, assigned_factory)
                actions[unit_id].extend(go_dig_rubble(
                    unit, game_state, self.position_registry, assigned_factory, factories_power,
                    self.rubble_tiles_being_dug, tiles_scores=None, n_min_digs=5, n_desired_digs=8,
                    tiles_for_assistants_only=assistant_tiles))
                        # break
            else:
                # if np.array_equal(unit.pos, assigned_factory.pos):
                #     actions[unit_id].extend([unit.move(1)])
                # actions[unit_id].extend([unit.recharge(x=3 * unit_cfg.DIG_COST)])
                # debug purposes only!  # todo: replace that self destruct
                actions[unit_id].extend([unit.self_destruct(repeat=0, n=1)])

            # if unit.unit_id in monitored_units and game_state.real_env_steps in monitored_turns:
            #     pass

            if game_state.env_steps in time_monitored_turns:
                end_time = time.time()
                local_time = str(end_time-start_time)
                pass

        if game_state.env_steps in overall_time_monitored_turns:
            end_time = time.time()
            local_time = str(end_time-start_time)
            pass

        # for u_id, action_queue in actions.items():
        # if game_state.real_env_steps >= 6:
        #     pass

        return actions
