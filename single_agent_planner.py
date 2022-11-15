import heapq

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3: Return a table that contains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.

    constraints_table = dict()

    for constraint in constraints:
        if constraint["agent"] == agent:
            if constraint["timestep"] not in constraints_table:
                constraints_table[constraint["timestep"]] = []

            constraints_table[constraint["timestep"]].append(constraint)

    return constraints_table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(child, constraint_table):
    ##############################
    # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.

    if constraint_table == {}:
        return False

    # -> If timestep has constraints
    if child["timestep"] in constraint_table.keys():  # or child["timestep"] > max(list(constraint_table.keys())):
        if child["timestep"] in constraint_table.keys():
            timestep_constraints = constraint_table[child["timestep"]]
        else:
            timestep_constraints = constraint_table[max(list(constraint_table.keys()))]

        # ... for every constraint
        for constraint in timestep_constraints:
            # if constraint is position constraint
            if isinstance(constraint["loc"], tuple):
                if constraint["loc"] == child["loc"]:
                    return True    # -> Child is constrained

            # if constraint is edge constraint
            elif isinstance(constraint["loc"], list):
                if child["parent"]["loc"] == constraint["loc"][0] and child["loc"] == constraint["loc"][1]:
                    return True    # -> Child is constrained

    return False    # -> Child is not constrained


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    # -> Construct constraint table
    constraint_table = build_constraint_table(constraints, agent)
    # print("> Constraint table:", constraint_table)

    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list_dict = dict()

    if constraint_table:
        earliest_goal_timestep = max(list(constraint_table.keys()))
    else:
        earliest_goal_timestep = 0

    # Current heuristics value given location
    h_value = h_values[start_loc]

    # -> Setup current node root
    root = {
        'loc': start_loc,
        'g_val': 0,
        'h_val': h_value,
        'parent': None,
        "timestep": 0
    }
    push_node(open_list, root)
    closed_list_dict[(root['loc'])] = root

    while len(open_list) > 0:

        curr = pop_node(open_list)
        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints
        # if curr['loc'] == goal_loc:
        #     return get_path(curr)

        if curr['loc'] == goal_loc and curr['timestep'] >= earliest_goal_timestep:
            return get_path(curr)

        # -> Check for child for all four directions (up, right, down, left, wait)
        for dir in range(5):
            child_loc = move(curr['loc'], dir)

            # Check if new location is not outside the map
            if child_loc[0] < 0 or child_loc[1] < 0 or child_loc[0] >= len(my_map) or child_loc[1] >= len(my_map[0]):
                continue

            # -> Check if new location has obstacle
            if my_map[child_loc[0]][child_loc[1]]:
                continue

            # -> Create new child node
            child = {
                'loc': child_loc,
                'g_val': curr['g_val'] + 1,
                'h_val': h_values[child_loc],
                'parent': curr,
                "timestep": curr["timestep"] + 1      # New timestep
            }

            # -> Check child clash with constraints
            if is_constrained(child, constraint_table):
                continue

            # -> Check if child node is in closed list
            if (child['loc'], child["timestep"]) in closed_list_dict:
                # -> Get existing node
                existing_node = closed_list_dict[(child['loc'], child["timestep"])]

                # -> Check if existing node is better than new node
                if compare_nodes(child, existing_node):
                    # -> Replace existing node with new node
                    closed_list_dict[(child['loc'], child["timestep"])] = child

                    # -> Push node to open list
                    push_node(open_list, child)

            else:
                # -> Add child node to closed list dict
                closed_list_dict[(child['loc'], child["timestep"])] = child

                # -> Push node to open list
                push_node(open_list, child)

        if curr["timestep"] > 100:
            return None

    return None  # Failed to find solutions
