import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from tabulate import tabulate
import numpy as np
import seaborn as sns


def get_df(ref):
    raw_df = pd.read_csv(ref)

    # Replace all run_time values of 0 with 0.0000000001
    raw_df['run_time'] = raw_df['run_time'].replace(0, 0.0000000001)

    # Creating copy of the raw dataframe
    df = deepcopy(raw_df)

    # --- Cost analysis
    # Add column with difference between cost and ideal cost
    df['total_cost_diff'] = df['cost'] - df['ideal_cost']

    # Normalise the cost difference per agent with respect to the number of agents
    df['cost_diff_per_agent'] = df['total_cost_diff'] / df['nb_agents']

    # --- Run time multiplied by cost
    df['run_time_per_cost_per_agent'] = (df['run_time'] / df['cost']) * df['nb_agents']

    df['efficiency'] = 10 ** (-(df['run_time_per_cost_per_agent'] * df['cost_diff_per_agent']))

    # Cleaning up the dataframe
    # del df['cost']
    del df['ideal_cost']
    del df['total_cost_diff']

    # Remove "instances/" from the test name column entries
    df['test_name'] = df['test_name'].str.replace('instances/', '')

    # Mean
    df_mean_by_agent_count = df.groupby('nb_agents', as_index=False).mean()

    # Cleanup dataframe
    del df_mean_by_agent_count["start_type"]

    df_mean_by_agent_count['run_time_per_cost_per_agent'] = (df_mean_by_agent_count['run_time'] / df_mean_by_agent_count['cost']) * df_mean_by_agent_count['nb_agents']
    df_mean_by_agent_count['efficiency'] = 10**(-(df_mean_by_agent_count['run_time_per_cost_per_agent']**2 * df_mean_by_agent_count['cost_diff_per_agent']))

    # Create a tile for the plot
    fig, ax = plt.subplots()

    # Set shape of plot to be a wide rectangle
    fig.set_size_inches(15, 5)

    # For every start type, create a heat map and tile plots
    for start_type in df['start_type'].unique():
        print("Start type: {}".format(start_type))
        # Create a dataframe for the current start type
        df_start_type = df[df['start_type'] == start_type]

        # Pivot the dataframe
        df_grouped_mean = df_start_type.pivot_table(
            index='test_name',
            columns='nb_agents',
            values='cost_diff_per_agent',
            aggfunc='mean')

        # Plot a heatmap with map type as y-axis and number of agents as x-axis, and the cost difference as the value in tile
        # Add plot to the tile according to the start type +1 (1, 2, 3)
        ax = plt.subplot(1, 3, start_type + 1)
        ax.set_title("Start type: {}".format(start_type))

        if start_type == 2:
            last_plot = True
        else:
            last_plot = False

        # Exclude label but include ticks for all but the first plot
        sns.heatmap(df_grouped_mean, annot=False, fmt=".2f", ax=ax, cmap="YlGnBu", cbar=last_plot)

        # Set the x-axis label bold
        ax.xaxis.label.set_weight("bold")
        # Set the y-axis label bold
        ax.yaxis.label.set_weight("bold")

    plt.show()

    return df_mean_by_agent_count


refs = ["results_Prioritized_v1.csv", "results_Distributed_v1.csv", "results_CBS_v1.csv"]

# Create a list of dataframes
dfs = [get_df(ref) for ref in refs]

# Plot all dataframes's efficiencies on the same plot
for df in dfs:
    plt.plot(df['nb_agents'], df['efficiency'])

plt.legend(['Prioritized', 'Distributed', 'CBS'])
plt.xlabel('Number of agents')
plt.ylabel('Efficiency')
plt.show()

# Plot all dataframes's cost diff per agent on the same plot
for df in dfs:
    plt.plot(df['nb_agents'], df['cost_diff_per_agent'])

plt.legend(['Prioritized', 'Distributed', 'CBS'])
plt.xlabel('Number of agents')
plt.ylabel('Cost difference per agent')
plt.show()