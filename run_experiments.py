"""
Main file to run experiments and show animation.

Note: To make the animation work in Spyder you should set graphics backend to 'Automatic' (Preferences > Graphics > Graphics Backend).
"""

#!/usr/bin/python
import argparse
import numpy as np
import glob
from pathlib import Path
from Solvers.cbs import CBSSolver
from Solvers.independent import IndependentSolver
from Solvers.prioritized import PrioritizedPlanningSolver
from Solvers.distributed import DistributedPlanningSolver # Placeholder for Distributed Planning
from visualize import Animation
from single_agent_planner import get_sum_of_cost

SOLVER = "CBS"


def print_mapf_instance(my_map, starts, goals):
    """
    Prints start location and goal location of all agents, using @ for an obstacle, . for a open cell, and 
    a number for the start location of each agent.
    
    Example:
        @ @ @ @ @ @ @ 
        @ 0 1 . . . @ 
        @ @ @ . @ @ @ 
        @ @ @ @ @ @ @ 
    """
    print('Start locations')
    print_locations(my_map, starts)
    print('Goal locations')
    print_locations(my_map, goals)


def print_locations(my_map, locations):
    """
    See docstring print_mapf_instance function above.
    """
    starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
    for i in range(len(locations)):
        starts_map[locations[i][0]][locations[i][1]] = i
    to_print = ''
    for x in range(len(my_map)):
        for y in range(len(my_map[0])):
            if starts_map[x][y] >= 0:
                to_print += str(starts_map[x][y]) + ' '
            elif my_map[x][y]:
                to_print += '@ '
            else:
                to_print += '. '
        to_print += '\n'
    print(to_print)


def import_mapf_instance(filename):
    """
    Imports mapf instance from instances folder. Expects input as a .txt file in the following format:
        Line1: #rows #columns (number of rows and columns)
        Line2-X: Grid of @ and . symbols with format #rows * #columns. The @ indicates an obstacle, whereas . indicates free cell.
        Line X: #agents (number of agents)
        Line X+1: xCoordStart yCoordStart xCoordGoal yCoordGoal (xy coordinate start and goal for Agent 1)
        Line X+2: xCoordStart yCoordStart xCoordGoal yCoordGoal (xy coordinate start and goal for Agent 2)
        Line X+n: xCoordStart yCoordStart xCoordGoal yCoordGoal (xy coordinate start and goal for Agent n)
        
    Example:
        4 7             # grid with 4 rows and 7 columns
        @ @ @ @ @ @ @   # example row with obstacle in every column
        @ . . . . . @   # example row with 5 free cells in the middle
        @ @ @ . @ @ @
        @ @ @ @ @ @ @
        2               # 2 agents in this experiment
        1 1 1 5         # agent 1 starts at (1,1) and has (1,5) as goal
        1 2 1 4         # agent 2 starts at (1,2) and has (1,4) as goal
    """
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # first line: #rows #columns
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    rows = int(rows)
    columns = int(columns)
    # #rows lines with the map
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    # #agents
    line = f.readline()
    num_agents = int(line)
    # #agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))
    f.close()
    return my_map, starts, goals

def get_agent_start_goal(my_map, starts, goals, reduced=True, start_left_side=True):
    # Pick a random start location
    if reduced:
        limit_start = 2
        limit_goal = len(my_map[0]) - 2
    else:
        limit_start = len(my_map[0])
        limit_goal = 0

    start = (np.random.randint(0, len(my_map)), np.random.randint(0, limit_start))

    # while start is not free
    while my_map[start[0]][start[1]] == 1 or start in starts:
        start = (np.random.randint(0, len(my_map)), np.random.randint(0, limit_start))

    # Pick a random goal location
    goal = (np.random.randint(0, len(my_map)), np.random.randint(limit_goal, len(my_map[0])))

    # while goal is not free
    while my_map[goal[0]][goal[1]] == 1 or goal in goals or goal == start:
        goal = (np.random.randint(0, len(my_map)), np.random.randint(limit_goal, len(my_map[0])))

    return start, goal


if __name__ == '__main__':
    print("Working")
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
    parser.add_argument('--instance', type=str, default=None,
                        help='The name of the instance file(s)')
    parser.add_argument('--batch', action='store_true', default=False,
                        help='Use batch output instead of animation')
    parser.add_argument('--disjoint', action='store_true', default=False,
                        help='Use the disjoint splitting')
    parser.add_argument('--solver', type=str, default=SOLVER,
                        help='The solver to use (one of: {CBS,Independent,Prioritized}), defaults to ' + str(SOLVER))
    parser.add_argument('--agent_count', type=int, default=3,
                        help='The number of agents to generate, defaults to ' + str(3))

    args = parser.parse_args()
    print(args)
    # Hint: Command line options can be added in Spyder by pressing CTRL + F6 > Command line options. 
    # In PyCharm, they can be added as parameters in the configuration.

    result_file = open("results.csv", "w", buffering=1)

    for file in sorted(glob.glob(args.instance)):
        print(file)

        print("***Import an instance***")
        my_map, starts, goals = import_mapf_instance(file)

        if not args.solver == "Distributed":
            print_mapf_instance(my_map, starts, goals)

        if args.solver == "CBS":
            starts = []
            goals = []

            # -> Generate start/goal pairs for the agents
            for i in range(args.agent_count):
                start, goal = get_agent_start_goal(my_map, starts, goals, reduced=True)
                starts.append(start)
                goals.append(goal)

            print_mapf_instance(my_map, starts, goals)
            print("***Run CBS***")
            cbs = CBSSolver(my_map, starts, goals)
            paths = cbs.find_solution(args.disjoint)

        elif args.solver == "Independent":
            print("***Run Independent***")
            solver = IndependentSolver(my_map, starts, goals)
            paths = solver.find_solution()

        elif args.solver == "Prioritized":
            starts = []
            goals = []

            # -> Generate start/goal pairs for the agents
            for i in range(args.agent_count):
                start, goal = get_agent_start_goal(my_map, starts, goals, reduced=True)
                starts.append(start)
                goals.append(goal)

            print_mapf_instance(my_map, starts, goals)
            print("***Run Prioritized***")
            solver = PrioritizedPlanningSolver(my_map, starts, goals)
            paths = solver.find_solution()

        elif args.solver == "Distributed":  # Wrapper of distributed planning solver class
            starts = []
            goals = []

            # -> Generate start/goal pairs for the agents
            for i in range(args.agent_count):
                start, goal = get_agent_start_goal(my_map, starts, goals, reduced=False)
                starts.append(start)
                goals.append(goal)

            print_mapf_instance(my_map, starts, goals)

            print("***Run Distributed Planning***")
            solver = DistributedPlanningSolver(my_map, starts, goals)
            paths = solver.find_solution()
        else:
            raise RuntimeError("Unknown solver!")

        cost = get_sum_of_cost(paths)
        result_file.write("{},{}\n".format(file, cost))

        if not args.batch:
            print("***Test paths on a simulation***")
            animation = Animation(my_map, starts, goals, paths)
            # animation.save("output.mp4", 1.0) # install ffmpeg package to use this option
            animation.show()
    result_file.close()
