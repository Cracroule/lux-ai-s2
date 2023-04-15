import numpy as np
import scipy
import sys
# from scipy.spatial.distance import cdist

from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory


def manhattan_dist_vect_point(loc1, loc2):
    return np.sum(np.abs(loc2 - loc1), 1)


def manhattan_dist_points(loc1, loc2):
    return np.sum(np.abs(loc2 - loc1))


# def rubble_weighted_density(rubble_array, loc):
#     np.average(a, axis, weights)


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

        self.bots = {}
        self.botpos = []
        self.bot_factory = {}
        self.factory_bots = {}
        self.factory_queue = {}
        self.move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

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

                min_dist = 10e6
                best_loc = potential_spawns[0]

                d_rubble = 10

                for loc in potential_spawns:

                    ice_tile_distances = np.mean((ice_tile_locations - loc) ** 2, 1)
                    ore_tile_distances = np.mean((ore_tile_locations - loc) ** 2, 1)
                    density_rubble = np.mean(
                        obs["board"]["rubble"][max(loc[0] - d_rubble, 0):min(loc[0] + d_rubble, 47),
                        max(loc[1] - d_rubble, 0):min(loc[1] + d_rubble, 47)])

                    closes_opp_factory_dist = 0
                    if len(opp_factories) >= 1:
                        closes_opp_factory_dist = np.min(np.mean((np.array(opp_factories) - loc) ** 2, 1))
                    closes_my_factory_dist = 0
                    if len(my_factories) >= 1:
                        closes_my_factory_dist = np.min(np.mean((np.array(my_factories) - loc) ** 2, 1))

                    minimum_ice_dist = np.min(ice_tile_distances) * 10 + 0.01 * np.min(
                        ore_tile_distances) + 10 * density_rubble / (
                                           d_rubble) - closes_opp_factory_dist * 0.1 + closes_my_factory_dist * 0.01

                    if minimum_ice_dist < min_dist:
                        min_dist = minimum_ice_dist
                        best_loc = loc

                #                 spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
                spawn_loc = best_loc
                actions['spawn'] = spawn_loc
                #                 actions['metal']=metal_left
                #                 actions['water']=water_left
                actions['metal'] = min(300, metal_left)
                actions['water'] = min(300, metal_left)

        return actions

    def check_collision(self, pos, direction, unitpos, unit_type='LIGHT'):
        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        #         move_deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])

        new_pos = pos + move_deltas[direction]

        if unit_type == "LIGHT":
            return str(new_pos) in unitpos or str(new_pos) in self.botposheavy.values()
        else:
            return str(new_pos) in unitpos

    def get_direction(self, unit, closest_tile, sorted_tiles):

        closest_tile = np.array(closest_tile)
        direction = direction_to(np.array(unit.pos), closest_tile)
        k = 0
        all_unit_positions = set(self.botpos.values())
        unit_type = unit.unit_type
        while self.check_collision(np.array(unit.pos), direction, all_unit_positions, unit_type) and k < min(
                len(sorted_tiles) - 1, 500):
            k += 1
            closest_tile = sorted_tiles[k]
            closest_tile = np.array(closest_tile)
            direction = direction_to(np.array(unit.pos), closest_tile)

        if self.check_collision(unit.pos, direction, all_unit_positions, unit_type):
            for direction_x in np.arange(4, -1, -1):
                if not self.check_collision(np.array(unit.pos), direction_x, all_unit_positions, unit_type):
                    direction = direction_x
                    break

        if self.check_collision(np.array(unit.pos), direction, all_unit_positions, unit_type):
            direction = np.random.choice(np.arange(5))

        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

        self.botpos[unit.unit_id] = str(np.array(unit.pos) + move_deltas[direction])

        return direction

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        1. Regular Phase
        2. Building Robots
        """
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        state_obs = obs

        # Unit locations
        self.botpos = {}
        self.botposheavy = {}
        self.opp_botpos = []
        for player in [self.player, self.opp_player]:
            for unit_id, unit in game_state.units[player].items():

                if player == self.player:
                    self.botpos[unit_id] = str(unit.pos)
                else:
                    self.opp_botpos.append(unit.pos)

                if unit.unit_type == "HEAVY":
                    self.botposheavy[unit_id] = str(unit.pos)

        # Build Robots
        factories = game_state.factories[self.player]
        factory_tiles, factory_units, factory_ids = [], [], []
        bot_units = {}

        for unit_id, factory in factories.items():

            if unit_id not in self.factory_bots.keys():
                self.factory_bots[unit_id] = {
                    'ice': [],
                    'ore': [],
                    'rubble': [],
                    'kill': [],
                }

                self.factory_queue[unit_id] = []

            for task in ['ice', 'ore', 'rubble', 'kill']:
                for bot_unit_id in self.factory_bots[unit_id][task]:
                    if bot_unit_id not in self.botpos.keys():
                        self.factory_bots[unit_id][task].remove(bot_unit_id)

            minbot_task = None
            min_bots = {
                'ice': 1,
                'ore': 5,
                'rubble': 5,
                'kill': 1
            }
            # NO. BOTS PER TASK
            for task in ['kill', 'ice', 'ore', 'rubble']:
                num_bots = len(self.factory_bots[unit_id][task]) + sum([task in self.factory_queue[unit_id]])
                if num_bots < min_bots[task]:
                    minbots = num_bots
                    minbot_task = task
                    break

            if minbot_task is not None:
                if minbot_task in ['kill', 'ice']:
                    if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                        actions[unit_id] = factory.build_heavy()
                    elif factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
                            factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST:
                        actions[unit_id] = factory.build_light()
                else:
                    if factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
                            factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST:
                        actions[unit_id] = factory.build_light()
                    elif factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                        actions[unit_id] = factory.build_heavy()

                if unit_id not in self.factory_queue.keys():
                    self.factory_queue[unit_id] = [minbot_task]
                else:
                    self.factory_queue[unit_id].append(minbot_task)

            factory_tiles += [factory.pos]
            factory_units += [factory]
            factory_ids += [unit_id]

            if factory.can_water(game_state) and step > 900 and factory.cargo.water > (1000 - step) + 100:
                actions[unit_id] = factory.water()

        factory_tiles = np.array(factory_tiles)  # Factory locations (to go back to)

        # Move Robots
        # iterate over our units and have them mine the closest ice tile
        units = game_state.units[self.player]

        # Resource map and locations
        ice_map = game_state.board.ice
        ore_map = game_state.board.ore
        rubble_map = game_state.board.rubble

        ice_locations_all = np.argwhere(ice_map >= 1)  # numpy position of every ice tile
        ore_locations_all = np.argwhere(ore_map >= 1)  # numpy position of every ore tile
        rubble_locations_all = np.argwhere(rubble_map >= 1)  # numpy position of every rubble tile

        ice_locations = ice_locations_all
        ore_locations = ore_locations_all
        rubble_locations = rubble_locations_all

        for unit_id, unit in iter(sorted(units.items())):

            if unit_id not in self.bots.keys():
                self.bots[unit_id] = ''

            if len(factory_tiles) > 0:
                closest_factory_tile = factory_tiles[0]

            if unit_id not in self.bot_factory.keys():
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                min_index = np.argmin(factory_distances)
                closest_factory_tile = factory_tiles[min_index]
                self.bot_factory[unit_id] = factory_ids[min_index]
            elif self.bot_factory[unit_id] not in factory_ids:
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                min_index = np.argmin(factory_distances)
                closest_factory_tile = factory_tiles[min_index]
                self.bot_factory[unit_id] = factory_ids[min_index]
            else:
                closest_factory_tile = factories[self.bot_factory[unit_id]].pos

            distance_to_factory = np.mean((np.array(closest_factory_tile) - np.array(unit.pos)) ** 2)
            adjacent_to_factory = False
            sorted_factory = [closest_factory_tile]

            if unit.power < unit.action_queue_cost(game_state):
                continue

            if len(factory_tiles) > 0:

                move_cost = None
                try:
                    adjacent_to_factory = np.mean((np.array(closest_factory_tile) - np.array(unit.pos)) ** 2) <= 1
                except:
                    print(closest_factory_tile, unit.pos)
                    assert False

                ## Assigning task for the bot
                if self.bots[unit_id] == '':
                    task = 'ice'
                    if len(self.factory_queue[self.bot_factory[unit_id]]) != 0:
                        task = self.factory_queue[self.bot_factory[unit_id]].pop(0)
                    self.bots[unit_id] = task
                    self.factory_bots[self.bot_factory[unit_id]][task].append(unit_id)

                battery_capacity = 150 if unit.unit_type == "LIGHT" else 3000
                cargo_space = 100 if unit.unit_type == "LIGHT" else 1000
                def_move_cost = 1 if unit.unit_type == "LIGHT" else 20
                rubble_dig_cost = 5 if unit.unit_type == "LIGHT" else 100

                if self.bots[unit_id] == "ice":
                    if unit.cargo.ice < cargo_space and unit.power > unit.action_queue_cost(game_state) + unit.dig_cost(
                            game_state) + def_move_cost * distance_to_factory:

                        # compute the distance to each ice tile from this unit and pick the closest

                        ice_rubbles = np.array([rubble_map[pos[0]][pos[1]] for pos in ice_locations])
                        ice_distances = np.mean((ice_locations - unit.pos) ** 2, 1)  # - (ice_rubbles)*10
                        sorted_ice = [ice_locations[k] for k in np.argsort(ice_distances)]

                        closest_ice = sorted_ice[0]
                        # if we have reached the ice tile, start mining if possible
                        if np.all(closest_ice == unit.pos):
                            if unit.power >= unit.dig_cost(game_state) + \
                                    unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.dig(repeat=False)]
                        else:
                            direction = self.get_direction(unit, closest_ice, sorted_ice)
                            move_cost = unit.move_cost(game_state, direction)

                    elif unit.cargo.ice >= cargo_space or unit.power <= unit.action_queue_cost(
                            game_state) + unit.dig_cost(game_state) + def_move_cost * distance_to_factory:

                        if adjacent_to_factory:
                            if unit.cargo.ice > 0:
                                actions[unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
                            elif unit.cargo.ore > 0:
                                actions[unit_id] = [unit.transfer(0, 1, unit.cargo.ore, repeat=False)]
                            elif unit.power < battery_capacity * 0.1:
                                actions[unit_id] = [unit.pickup(4, battery_capacity - unit.power)]
                        else:
                            direction = self.get_direction(unit, closest_factory_tile, sorted_factory)
                            move_cost = unit.move_cost(game_state, direction)

                elif self.bots[unit_id] == 'ore':
                    if unit.cargo.ore < cargo_space and unit.power > unit.action_queue_cost(game_state) + unit.dig_cost(
                            game_state) + def_move_cost * distance_to_factory:

                        # compute the distance to each ore tile from this unit and pick the closest
                        ore_rubbles = np.array([rubble_map[pos[0]][pos[1]] for pos in ore_locations])
                        ore_distances = np.mean((ore_locations - unit.pos) ** 2, 1)  # + (ore_rubbles)*2
                        sorted_ore = [ore_locations[k] for k in np.argsort(ore_distances)]

                        closest_ore = sorted_ore[0]
                        # if we have reached the ore tile, start mining if possible
                        if np.all(closest_ore == unit.pos):
                            if unit.power >= unit.dig_cost(game_state) + \
                                    unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.dig(repeat=False)]
                        else:
                            direction = self.get_direction(unit, closest_ore, sorted_ore)
                            move_cost = unit.move_cost(game_state, direction)

                    elif unit.cargo.ore >= cargo_space or unit.power <= unit.action_queue_cost(
                            game_state) + unit.dig_cost(game_state) + def_move_cost * distance_to_factory:

                        if adjacent_to_factory:
                            if unit.cargo.ore > 0:
                                actions[unit_id] = [unit.transfer(0, 1, unit.cargo.ore, repeat=False)]
                            elif unit.cargo.ice > 0:
                                actions[unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
                            elif unit.power < battery_capacity * 0.1:
                                actions[unit_id] = [unit.pickup(4, battery_capacity - unit.power)]
                        else:
                            direction = self.get_direction(unit, closest_factory_tile, sorted_factory)
                            move_cost = unit.move_cost(game_state, direction)
                elif self.bots[unit_id] == 'rubble':
                    if unit.power > unit.action_queue_cost(game_state) + unit.dig_cost(game_state) + rubble_dig_cost:

                        # compute the distance to each rubble tile from this unit and pick the closest
                        rubble_distances = np.mean((rubble_locations - unit.pos) ** 2, 1)
                        sorted_rubble = [rubble_locations[k] for k in np.argsort(rubble_distances)]
                        closest_rubble = sorted_rubble[0]

                        # if we have reached the rubble tile, start mining if possible
                        if np.all(closest_rubble == unit.pos) or rubble_map[unit.pos[0], unit.pos[1]] != 0:
                            if unit.power >= unit.dig_cost(game_state) + \
                                    unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.dig(repeat=False)]
                        else:
                            if len(rubble_locations) != 0:
                                direction = self.get_direction(unit, closest_rubble, sorted_rubble)
                                move_cost = unit.move_cost(game_state, direction)

                    elif unit.power <= unit.action_queue_cost(game_state) + unit.dig_cost(game_state) + rubble_dig_cost:

                        if adjacent_to_factory:
                            if unit.cargo.ore > 0:
                                actions[unit_id] = [unit.transfer(0, 1, unit.cargo.ore, repeat=False)]
                            elif unit.cargo.ice > 0:
                                actions[unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
                            elif unit.power < battery_capacity * 0.1:
                                actions[unit_id] = [unit.pickup(4, battery_capacity - unit.power)]
                        else:
                            direction = self.get_direction(unit, closest_factory_tile, sorted_factory)
                            move_cost = unit.move_cost(game_state, direction)
                elif self.bots[unit_id] == 'kill':

                    if len(self.opp_botpos) != 0:
                        opp_pos = np.array(self.opp_botpos).reshape(-1, 2)
                        opponent_unit_distances = np.mean((opp_pos - unit.pos) ** 2, 1)
                        min_distance = np.min(opponent_unit_distances)
                        pos_min_distance = opp_pos[np.argmin(min_distance)]

                        if min_distance == 1:
                            direction = self.get_direction(unit, np.array(pos_min_distance),
                                                           [np.array(pos_min_distance)])
                            move_cost = unit.move_cost(game_state, direction)
                        else:
                            if unit.power > unit.action_queue_cost(game_state):
                                direction = self.get_direction(unit, np.array(pos_min_distance),
                                                               [np.array(pos_min_distance)])
                                move_cost = unit.move_cost(game_state, direction)
                            else:
                                if adjacent_to_factory:
                                    if unit.cargo.ore > 0:
                                        actions[unit_id] = [unit.transfer(0, 1, unit.cargo.ore, repeat=False)]
                                    elif unit.cargo.ice > 0:
                                        actions[unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
                                    elif unit.power < battery_capacity * 0.1:
                                        actions[unit_id] = [unit.pickup(4, battery_capacity - unit.power)]
                                else:
                                    direction = self.get_direction(unit, closest_factory_tile, sorted_factory)
                                    move_cost = unit.move_cost(game_state, direction)

                # check move_cost is not None, meaning that direction is not blocked
                # check if unit has enough power to move and update the action queue.
                if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                    actions[unit_id] = [unit.move(direction, repeat=False)]

        return actions