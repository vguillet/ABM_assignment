"""
This file contains the AircraftDistributed class that can be used to implement individual planning.

Code in this file is just provided as guidance, you are free to deviate from it.
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class AircraftDistributed(object):
    """Aircraft object to be used in the distributed planner."""

    def __init__(self, my_map, start, goal, heuristics, agent_id):
        """
        my_map   - list of lists specifying obstacle positions
        starts      - (x1, y1) start location
        goals       - (x1, y1) goal location
        heuristics  - heuristic to goal location
        """

        self.obstacle_map = deepcopy(np.array(my_map))
        self.my_weights = np.ones((len(my_map), len(my_map[0])))
        self.start = start
        self.goal = goal
        self.id = agent_id
        self.heuristics = heuristics    # Also known as h_values

        self.loc = start    # Current location
        self.path = [start]
        print("Goal of agent", self.id, "is", self.goal)

    def step(self, agents_location_map, agents_states_dict):
        # -> Get available actions
        print("Agent_ID analysed", self.id)
        available_actions = self.get_available_actions(agents_location_map=agents_location_map)

        # -> Agents in visibility radius
        agents_in_visibility_radius = []

        # -> Compute distance vector to each agent
        for agent_id, agent_state in agents_states_dict.items():
            distance_vector = (agent_state["loc"][0] - self.loc[0], agent_state["loc"][1] - self.loc[1])
            distance_magnitude = np.sqrt(distance_vector[0] ** 2 + distance_vector[1] ** 2)

            if distance_magnitude < 5 and agent_id != self.id and agent_state["ideal_path_to_goal"]:
                agents_in_visibility_radius.append(agent_id)

        # -> Compute repulsive forces
        repulsive_forces = np.zeros((len(self.my_weights), len(self.my_weights[0])))

        # y = np.linspace(0, 9, 9)
        # x = np.linspace(0, 22, 22)
        #
        # xx, yy = np.meshgrid(x, y)
        #
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(xx, yy, repulsive_forces, cmap='viridis', linewidth=0)
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')
        # plt.show()

        for agent_id in agents_in_visibility_radius:
            agent_state = agents_states_dict[agent_id]
            # -> Add repulsive weight centered around agent location
            repulsive_forces += \
                self.gen_repulsive_field(loc=agent_state["loc"]) * 1/len(agent_state["ideal_path_to_goal"]) * 1.3

        # -> Normalize array between 0 and 1
        if agents_in_visibility_radius:
            repulsive_forces = \
                (repulsive_forces - np.amin(repulsive_forces)) / (np.amax(repulsive_forces) - np.amin(repulsive_forces))

        # -> Compute each action's cost
        costs = []

        # print("Repulsive force", repulsive_forces)

        for action in available_actions:
            costs.append(self.heuristics[action] * self.my_weights[action] + repulsive_forces[action])

        # print("Costs", costs)

        # -> Choose action with lowest cost
        action = available_actions[costs.index(min(costs))]

        # -> Update weight map
        # Forget
        self.my_weights = self.my_weights - 0.05

        # Decrease prev loc appeal
        self.my_weights[self.loc[0]][self.loc[1]] += 0.3

        # -> Update agent's location
        self.loc = action

        # -> Update agent's path
        self.path.append(action)

    def gen_repulsive_field(self, loc):
        # --> Getting max shape dimension
        if self.obstacle_map.shape[0] > self.obstacle_map.shape[1]:
            max_dim = self.obstacle_map.shape[0]
            min_dim = self.obstacle_map.shape[1]
            min_axis = 1
        else:
            max_dim = self.obstacle_map.shape[1]
            min_dim = self.obstacle_map.shape[0]
            min_axis = 0

        x = np.linspace(0, max_dim, max_dim)
        y = np.linspace(0, max_dim, max_dim)

        xx, yy = np.meshgrid(x, y)

        grid = -np.sqrt((xx - loc[0]) ** 2 +
                        (yy - loc[1]) ** 2)

        grid += np.amax(np.absolute(grid))

        # -> Normalize array between 0 and 1
        grid = (grid - np.amin(grid)) / (np.amax(grid) - np.amin(grid))

        # # --> Make a 3D plot
        if min_axis == 0:
            return grid[0:min_dim, :]
        else:
            return grid[:, 0:min_dim]

    def get_available_actions(self, loc=None, agents_location_map=None, wait=True):
        """
        Returns a list of available actions (new locations) for the agent.
        """
        # print("loc_before", loc)
        if loc is None:
            loc = self.loc

        # print("loc_after", loc)

        # (up, right, down, left, wait)
        if wait:
            actions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
        else:
            actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        available_actions = []

        # -> Check for each action if it is available
        for action in actions:
            # -> Compute new location
            new_loc = (loc[0] + action[0], loc[1] + action[1])

            # -> Check if new location is valid
            # print("Special print", self.obstacle_map, type(self.obstacle_map))
            # print("new_loc", new_loc)
            if 0 <= new_loc[0] < self.obstacle_map.shape[0] and 0 <= new_loc[1] < self.obstacle_map.shape[1]:
                # -> Check if new location is free
                # if action == (0, 0) or agents_location_map[new_loc[0]][new_loc[1]] == 0 and self.obstacle_map[new_loc[0]][new_loc[1]] == 0:
                #     available_actions.append(new_loc)

                if self.obstacle_map[new_loc[0]][new_loc[1]] == 0:
                    if agents_location_map is not None:
                        if agents_location_map[new_loc[0]][new_loc[1]] == 0:
                            condition = True
                        else:
                            condition = False
                    else:
                        condition = True
                else:
                    condition = False

                if action == (0, 0) or condition:
                    available_actions.append(new_loc)

        return available_actions

    @property
    def at_goal(self) -> bool:
        return self.loc == self.goal

    @property
    def ideal_path_to_goal(self) -> list:
        virtual_loc = deepcopy(self.loc)
        path = [virtual_loc]

        while virtual_loc != self.goal:
            # -> Get available actions
            available_actions = self.get_available_actions(loc=virtual_loc, wait=False)
            # print("agent:", self.id, "available_actions", available_actions, ", virtual loc", virtual_loc, self.goal)

            # -> Compute each action's cost
            costs = []

            for action in available_actions:
                costs.append(self.heuristics[action])

            # -> Get action with the lowest cost
            action = available_actions[costs.index(min(costs))]

            # print("action", available_actions, self.id, virtual_loc)

            # -> Update virtual location
            virtual_loc = action

            # -> Add action to path
            path.append(action)

        return path
