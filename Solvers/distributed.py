"""
This file contains a placeholder for the DistributedPlanningSolver class that can be used to implement distributed planning.

Code in this file is just provided as guidance, you are free to deviate from it.
"""

import time as timer
from single_agent_planner import get_sum_of_cost, compute_heuristics, a_star
from aircraft import AircraftDistributed
import numpy as np
from copy import deepcopy


class DistributedPlanningSolver(object):
    """A distributed planner"""

    def __init__(self, my_map, starts, goals):
        """
        my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """
        print(starts)
        print(goals)
        self.CPU_time = 0
        self.my_map = my_map
        self.my_map_shadow = deepcopy(my_map)
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.agents = []        # List of agent objects

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def get_agents_location_map(self):
        """
        Returns the map state for all agents.
        """

        # -> Create empty array with size of map
        map_state = np.zeros((len(self.my_map), len(self.my_map[0])))

        # -> Add the location of each agent as a 1 in the array
        for agent in self.agents:
            map_state[agent.loc] = 1

        return map_state

    @staticmethod
    def update_agent_state(agent):
        """
        Used to get dict of agents and various extra states
        """
        agents_state = {
            "loc": agent.loc
        }

        if agent.at_goal:
            agents_state["ideal_path_to_goal"] = []

        else:
            agents_state["ideal_path_to_goal"] = agent.ideal_path_to_goal

        return agents_state

    def find_solution(self):
        """
        Finds paths for all agents from start to goal locations. 
        
        Returns:
            result (list): with a path [(s,t), .....] for each agent.
        """

        ideal_result = []
        for i in range(self.num_of_agents):  # Find path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, [])
            if path is None:
                raise BaseException('No solutions')

            ideal_result.append(path)

        # -> Initialize constants
        start_time = timer.time()

        result = []

        # -> Create agent objects with AircraftDistributed class
        for i in range(self.num_of_agents):
            # -> Compute heuristics for new agent
            heuristics = compute_heuristics(my_map=deepcopy(self.my_map), goal=self.goals[i])

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
        epoch_cap = 100
        epoch_count = 0

        agents_states_dict = {}

        for agent in self.agents:
            # -> Get agent states
            agents_states_dict[agent.id] = {
                "ideal_path_to_goal": agent.ideal_path_to_goal,
                "loc": agent.loc
            }

        while not all(newAgent.at_goal for newAgent in self.agents) and epoch_count < epoch_cap:
            epoch_count += 1

            print(epoch_count)

            for agent in self.agents:
                # If agent has not reached goal
                if not agent.at_goal:
                    # -> Get map state
                    agents_location_map = self.get_agents_location_map()

                    # -> Call step function
                    agent.step(
                        agents_location_map=agents_location_map,
                        agents_states_dict=agents_states_dict
                    )

                    # -> Update agent state
                    agents_states_dict[agent.id] = self.update_agent_state(agent=agent)

                    if agent.at_goal:
                        # print("Agent", agent.id, "at goal")
                        # -> Add agent location as permanent obstacle in my_map
                        self.my_map_shadow[agent.loc[0]][agent.loc[1]] = True
                        # print(self.my_map_shadow)

                        for other_agent in self.agents:
                            if not other_agent.at_goal:
                                # print(other_agent.id)
                                # -> Re-compute heuristics for all agents not at goal
                                other_agent.obstacle_map = np.array(self.my_map_shadow)
                                other_agent.heuristics = compute_heuristics(my_map=self.my_map_shadow, goal=other_agent.goal)

                                # -> Reset weights
                                other_agent.my_weights = np.ones((len(self.my_map), len(self.my_map[0])))

                                # -> Update agent state
                                agents_states_dict[other_agent.id] = self.update_agent_state(agent=other_agent)

        for agent in self.agents:
            result.append(agent.path)

        # Print final output
        self.CPU_time = timer.time() - start_time
        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))  # Hint: think about how cost is defined in your implementation
        print(result)
        
        return result, ideal_result, self.CPU_time  # Hint: this should be the final result of the distributed planning (visualization is done after planning)
