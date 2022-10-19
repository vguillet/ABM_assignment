"""
This file contains a placeholder for the DistributedPlanningSolver class that can be used to implement distributed planning.

Code in this file is just provided as guidance, you are free to deviate from it.
"""

import time as timer
from single_agent_planner import get_sum_of_cost, compute_heuristics
from aircraft import AircraftDistributed
import numpy as np
from copy import deepcopy


class DistributedPlanningSolver(object):
    """A distributed planner"""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """
        self.CPU_time = 0
        self.my_map = my_map
        self.my_map_shadow = deepcopy(my_map)
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.agents = []        # List of agent objects

    def get_map_state(self):
        """
        Returns the map state for all agents.
        """

        # -> Create empty array with size of map
        map_state = np.zeros((len(self.my_map), len(self.my_map[0])))

        # -> Add the location of each agent as a 1 in the array
        for agent in self.agents:
            map_state[agent.loc] = 1

        return map_state
        
    def find_solution(self):
        """
        Finds paths for all agents from start to goal locations. 
        
        Returns:
            result (list): with a path [(s,t), .....] for each agent.
        """
        # -> Initialize constants
        start_time = timer.time()
        result = []
        self.CPU_time = timer.time() - start_time

        # -> Create agent objects with AircraftDistributed class
        for i in range(self.num_of_agents):
            # -> Compute heuristics for new agent
            heuristics = compute_heuristics(my_map=self.my_map, goal=self.goals[i])

            # -> Create new agent object
            newAgent = AircraftDistributed(
                my_map=deepcopy(self.my_map),
                start=self.starts[i],
                goal=self.goals[i],
                heuristics=heuristics,
                agent_id=i
            )

            # -> Add agent to list of agents
            self.agents.append(newAgent)

        # While all agents have not reached their goal
        epoch_cap = 1000
        epoch_count = 0

        while not all(newAgent.at_goal for newAgent in self.agents) and epoch_count < epoch_cap:
            epoch_count += 1

            for agent in self.agents:
                # If agent has not reached goal
                if not agent.at_goal:
                    # -> Get map state
                    map_state = self.get_map_state()

                    # -> Call step function
                    agent.step(map_state=map_state)

                    if agent.at_goal:
                        # -> Add agent location as permanent obstacle in my_map
                        self.my_map_shadow[agent.loc[0]][agent.loc[1]] = 1

                        for agent in self.agents:
                            if not agent.at_goal:
                                # -> Re-compute heuristics for all agents not at goal
                                agent.heuristics = compute_heuristics(my_map=self.my_map_shadow, goal=agent.goal)

                                # -> Reset weights
                                agent.my_weights = np.ones((len(self.my_map), len(self.my_map[0])))

        for agent in self.agents:
            result.append(agent.path)

        # Print final output
        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))  # Hint: think about how cost is defined in your implementation
        print(result)
        
        return result  # Hint: this should be the final result of the distributed planning (visualization is done after planning)