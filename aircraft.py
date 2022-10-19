"""
This file contains the AircraftDistributed class that can be used to implement individual planning.

Code in this file is just provided as guidance, you are free to deviate from it.
"""

import numpy as np

class AircraftDistributed(object):
    """Aircraft object to be used in the distributed planner."""

    def __init__(self, my_map, start, goal, heuristics, agent_id):
        """
        my_map   - list of lists specifying obstacle positions
        starts      - (x1, y1) start location
        goals       - (x1, y1) goal location
        heuristics  - heuristic to goal location
        """

        self.my_map = my_map
        self.my_weights = np.ones((len(my_map), len(my_map[0])))
        self.start = start
        self.goal = goal
        self.id = agent_id
        self.heuristics = heuristics    # Also known as h_values

        self.loc = start    # Current location
        self.path = []

    def step(self, map_state):
        # -> Get available actions
        available_actions = self.get_available_actions(map_state)

        # -> Compute each action's cost
        costs = []

        for action in available_actions:
            costs.append(self.heuristics[action] * self.my_weights[action])

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

    def get_available_actions(self, map_state):
        """
        Returns a list of available actions (new locations) for the agent.
        """

        # (up, right, down, left, wait)
        actions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
        # actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        available_actions = []

        # -> Check for each action if it is available
        for action in actions:
            # -> Compute new location
            new_loc = (self.loc[0] + action[0], self.loc[1] + action[1])

            # -> Check if new location is valid
            if 0 <= new_loc[0] < map_state.shape[0] and 0 <= new_loc[1] < map_state.shape[1]:
                # -> Check if new location is free
                if action == (0, 0) or map_state[new_loc[0]][new_loc[1]] == 0 and self.my_map[new_loc[0]][new_loc[1]] == 0:
                    available_actions.append(new_loc)

        return available_actions

    @property
    def at_goal(self) -> bool:
        return self.loc == self.goal
