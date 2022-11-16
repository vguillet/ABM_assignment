import time as timer
import heapq
import random
from ABM_assignment.single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from copy import deepcopy


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    # -> Iterate through all timesteps and check for collisions
    path_max = max(len(path1), len(path2))

    for step in range(path_max):
        # -> Check if the robots are at the same location
        if get_location(path1, step) == get_location(path2, step):
            return {'type': 'vertex', 'loc': get_location(path1, step), 'timestep': step}

    path_min = min(len(path1), len(path2))

    for step in range(path_min):
        # -> Check if the robots swap their location
        if get_location(path1, step) == get_location(path2, step + 1) and get_location(path1, step + 1) == get_location(path2, step):
            return {'type': 'edge', 'loc': [get_location(path1, step), get_location(path1, step + 1)], 'timestep': step+1}

    return None


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    # -> Check collisions between all robot path pairs
    collisions = []

    for i in range(len(paths) - 1):     # ... for every agent
        for j in range(i + 1, len(paths)):      # ... check every other agent
            collision = detect_collision(paths[i], paths[j])
            if collision is not None:
                # -> Add robot ids to collision
                collision['a1'] = i
                collision['a2'] = j

                # -> Add collision to list
                collisions.append(collision)

    return collisions


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    if collision['type'] == 'vertex':
        return [{'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']},
                {'agent': collision['a2'], 'loc': collision['loc'], 'timestep': collision['timestep']}]
    else:  # Edge constraint (the second agent gets the limits switched)
        return [{'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']},
                {'agent': collision['a2'], 'loc': [collision['loc'][1], collision['loc'][0]], 'timestep': collision['timestep']}]


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly

    pass


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        self.num_of_expanded += 1
        return node

    @staticmethod
    def check_paths_equality(path1, path2):
        for i in range(len(path1)):  # for each agent
            if len(path1[i]) != len(path2[i]):  # if the length of their paths is different return False
                return False
            for j in range(len(path1[i])):  # if the paths are different return False
                if path1[i][j] != path2[i][j]:
                    return False
        return True

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root_vals = root['paths'].copy()

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # Task 3.1: Testing
        # print(root['collisions'])
        #
        # # Task 3.2: Testing
        # for collision in root['collisions']:
        #     print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit
        tick = 1
        best_node = None
        default_iter = 3000
        iter_counter = default_iter

        paths_studied = []

        while len(self.open_list) > 0:  # As long as there are still nodes in the open_list
            # -> Home made stuff
            tick += 1
            if tick % 50 == 0:
                # print("Length of open list", len(self.open_list))#, "\nCost of current node", current_node['cost'])
                # print("Current cost", current_node['cost'], len(current_node['collisions']))
                tick = 1

                # if best_node is not None:
                #      print("--> Current best node:", best_node['cost'])

            # Termination conditions
            if best_node is not None:
                iter_counter -= 1
                if iter_counter == 0:
                    # If there are no collisions, return the solution
                    CPU_time = self.print_results(root)
                    return best_node['paths'], root_vals, CPU_time

            current_node = self.pop_node()  # Pop node from the list with smallest cost
            paths_studied.append(current_node['paths'])

            if len(current_node['collisions']) != 0:  # If there are collisions
                for existing_collision in current_node['collisions']:   # For each collision
                    new_constraints = standard_splitting(existing_collision)  # Get collisions and split them in constraints for agents

                    for new_constraint in new_constraints:
                        # Add new constraint if it is not already in the list
                        new_node_constraints = current_node['constraints'].copy()

                        # if new_constraint not in current_node['constraints']:  # Make new node for each agent's perspective
                        #     new_node_constraints.append(new_constraint)

                        new_node_constraints.append(new_constraint)

                        new_node_Q = {
                            'cost': 0,      # Placeholder
                            'constraints': new_node_constraints,
                            'paths': current_node['paths'].copy(),
                            'collisions': []
                        }

                        # Find the new path of the new constraint's agent
                        agent_i = new_constraint['agent']

                        path = a_star(
                            my_map=self.my_map,
                            start_loc=self.starts[agent_i],
                            goal_loc=self.goals[agent_i],
                            h_values=self.heuristics[agent_i],
                            agent=agent_i,
                            constraints=new_node_Q['constraints']
                        )

                        if path is not None:
                            # If you could find a path, detect if there are again collisions and then push the node to the list
                            new_node_Q['paths'][agent_i] = path
                            new_node_Q['collisions'] = detect_collisions(new_node_Q['paths'])
                            new_node_Q['cost'] = get_sum_of_cost(new_node_Q['paths'])

                            # self.push_node(new_node_Q)

                            # If open list is empty (beginning iteration)
                            if not self.open_list:
                                self.push_node(new_node_Q)

                            elif new_node_Q['cost'] > root['cost'] + 10:
                                pass

                            # elif len(new_node_Q['collisions']) > root['cost'] + 3:
                            #     pass

                            elif len(new_node_Q['collisions']) > len(current_node['collisions']) + 1:
                                pass

                            elif len(new_node_Q['collisions']) > 5:
                                pass

                            else:
                                # Check if new path already explored
                                for path in paths_studied:
                                    if self.check_paths_equality(new_node_Q['paths'], path):
                                        break
                                else:
                                    self.push_node(new_node_Q)

                            if len(new_node_Q['collisions']) == 0:
                                if best_node is not None:
                                    if new_node_Q['cost'] < best_node['cost']:
                                        best_node = new_node_Q
                                        iter_counter = default_iter
                                else:
                                    best_node = new_node_Q
                                    iter_counter = default_iter

            else:
                # If there are no collisions, return the solution
                CPU_time = self.print_results(current_node)
                return current_node['paths'], root_vals, CPU_time

        CPU_time = self.print_results(root)
        return root['paths'], root_vals, CPU_time

    def print_results(self, node):
        # print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        # print("CPU time (s):    {:.2f}".format(CPU_time))
        # print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        # print("Expanded nodes:  {}".format(self.num_of_expanded))
        # print("Generated nodes: {}".format(self.num_of_generated))
        return CPU_time