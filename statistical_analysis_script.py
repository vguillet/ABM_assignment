import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results_Prioritized.csv')

# ------------------------ Derive data ----------------------------------------
# Add a new column to the dataframe normalising cost with the agent_count
df['cost_per_agent'] = df['cost'] / df['agent_count']

# Add column with difference between cost and ideal cost
df['cost_diff'] = df['cost'] - df['ideal_cost']

# Derive a column with the cost difference per agent
df['cost_diff_per_agent'] = df['cost_diff'] / df['agent_count']

# Divide the cost_per_agent by the avg_deviation to get the ideal cost
# df['ideal_cost'] = df['cost_per_agent'] / df['avg_deviation']

# Add a new column to the dataframe normalising avg_deviation with the agent_count
df['avg_deviation_per_agent'] = df['avg_deviation'] / df['agent_count']

print(df.to_string())

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
mean_total_avg_deviation_per_agent_count = df.groupby('agent_count')['avg_deviation'].mean()
print(f'Mean total avg deviation per agent count: {mean_total_avg_deviation_per_agent_count}')

# Std deviation for each agent count
std_total_avg_deviation_per_agent_count = df.groupby('agent_count')['avg_deviation'].std()
print(f'Std total avg deviation per agent count: {std_total_avg_deviation_per_agent_count}')

# Plot the average deviation per agent count
mean_total_avg_deviation_per_agent_count.plot(kind='bar')
plt.show()

# Avg deviation per agent for each agent count
mean_avg_deviation_per_agent_per_agent_count = df.groupby('agent_count')['avg_deviation_per_agent'].mean()
print(f'Mean avg deviation per agent per agent count: {mean_avg_deviation_per_agent_per_agent_count}')

# Std deviation per agent for each agent count
std_avg_deviation_per_agent_per_agent_count = df.groupby('agent_count')['avg_deviation_per_agent'].std()
print(f'Std avg deviation per agent per agent count: {std_avg_deviation_per_agent_per_agent_count}')

# Plot the average deviation per agent count
mean_avg_deviation_per_agent_per_agent_count.plot(kind='bar')
plt.show()

# Correlation of the cost with the agent count
corr_cost_agent_count = df['cost'].corr(df['agent_count'])
print(f'Correlation of the cost with the agent count: {corr_cost_agent_count}')

# Correlation of the total avg deviation with the agent count
corr_total_avg_deviation_agent_count = df['avg_deviation'].corr(df['agent_count'])
print(f'Correlation of the total avg deviation with the agent count: {corr_total_avg_deviation_agent_count}')

# Correlation of the avg deviation per agent with the agent count
corr_total_avg_deviation_per_agent_agent_count = df['avg_deviation_per_agent'].corr(df['agent_count'])
print(f'Correlation of the avg deviation per agent with the agent count: {corr_total_avg_deviation_per_agent_agent_count}')

# Population variance of the cost diff
var_cost_diff = df['cost_diff'].var()
print(f'Population variance of the cost diff: {var_cost_diff}')

