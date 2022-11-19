import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results_CBS.csv')

# ------------------------ Derive data ----------------------------------------
# Add a new column to the dataframe normalising cost with the nb_agents
df['cost_per_agent'] = df['cost'] / df['nb_agents']

# Add column with difference between cost and ideal cost
df['cost_diff'] = df['cost'] - df['ideal_cost']

# cost_diff for each agent count
cost_diff_per_nb_agents = df.groupby('nb_agents')['cost_diff'].mean()
print(f'mean cost_diff per agent count: {cost_diff_per_nb_agents}')

# Plot the average deviation per agent count
cost_diff_per_nb_agents.plot(kind='bar')
plt.show()

# Derive a column with the cost difference per agent
df['cost_diff_per_agent'] = df['cost_diff'] / df['nb_agents']

# Divide the cost_per_agent by the avg_deviation to get the ideal cost
# df['ideal_cost'] = df['cost_per_agent'] / df['avg_deviation']

# Add a new column to the dataframe normalising avg_deviation with the nb_agents
df['avg_deviation_per_agent'] = df['avg_deviation'] / df['nb_agents']

# print(df.to_string())
print(df.keys())

# ------------------------ Statistics -----------------------------------------
# Mean cost diff
mean_cost_diff = df['cost_diff'].mean()
print(f'Mean cost diff: {mean_cost_diff}')

# Standard deviation of cost diff\
std_cost_diff = df['cost_diff'].std()
print(f'Std cost diff: {std_cost_diff}')

# Mean cost diff per agent
mean_cost_diff_per_agent = df['cost_diff_per_agent'].mean()

# Standard deviation of cost diff per agent
std_cost_diff_per_agent = df['cost_diff_per_agent'].std()
print(f'Std cost diff per agent: {std_cost_diff_per_agent}')

# Mean total avg deviation
mean_total_avg_deviation = df['avg_deviation'].mean()
print(f'Mean total avg deviation: {mean_total_avg_deviation}')

# Standard deviation of total avg deviation
std_total_avg_deviation = df['avg_deviation'].std()
print(f'Std total avg deviation: {std_total_avg_deviation}')

# Mean avg deviation per agent
mean_avg_deviation_per_agent = df['avg_deviation_per_agent'].mean()
print(f'Mean avg deviation per agent: {mean_avg_deviation_per_agent}')

# Standard deviation of avg deviation per agent
std_avg_deviation_per_agent = df['avg_deviation_per_agent'].std()
print(f'Std avg deviation per agent: {std_avg_deviation_per_agent}')

# Avg deviation for each agent count
mean_total_avg_deviation_per_nb_agents = df.groupby('nb_agents')['avg_deviation'].mean()
print(f'Mean total avg deviation per agent count: {mean_total_avg_deviation_per_nb_agents}')

# Std deviation for each agent count
std_total_avg_deviation_per_nb_agents = df.groupby('nb_agents')['avg_deviation'].mean()
print(f'mean total avg deviation per agent count: {std_total_avg_deviation_per_nb_agents}')

# Plot the average deviation per agent count
mean_total_avg_deviation_per_nb_agents.plot(kind='bar', ylabel='mean of the average deviation')
plt.show()

# Avg deviation per agent for each agent count
mean_avg_deviation_per_agent_per_nb_agents = df.groupby('nb_agents')['avg_deviation_per_agent'].mean()
print(f'Mean avg deviation per agent per agent count: {mean_avg_deviation_per_agent_per_nb_agents}')

# Std deviation per agent for each agent count
std_avg_deviation_per_agent_per_nb_agents = df.groupby('nb_agents')['avg_deviation_per_agent'].mean()
print(f'avg deviation per agent per agent count: {std_avg_deviation_per_agent_per_nb_agents}')

# Plot the average deviation per agent count
mean_avg_deviation_per_agent_per_nb_agents.plot(kind='bar')
plt.show()

# Correlation of the cost with the agent count
corr_cost_nb_agents = df['cost'].corr(df['nb_agents'])
print(f'Correlation of the cost with the agent count: {corr_cost_nb_agents}')

# Correlation of the total avg deviation with the agent count
corr_total_avg_deviation_nb_agents = df['avg_deviation'].corr(df['nb_agents'])
print(f'Correlation of the total avg deviation with the agent count: {corr_total_avg_deviation_nb_agents}')

# Correlation of the avg deviation per agent with the agent count
corr_total_avg_deviation_per_agent_nb_agents = df['avg_deviation_per_agent'].corr(df['nb_agents'])
print(f'Correlation of the avg deviation per agent with the agent count: {corr_total_avg_deviation_per_agent_nb_agents}')

# Population variance of the cost diff
var_cost_diff = df['cost_diff'].var()
print(f'Population variance of the cost diff: {var_cost_diff}')

