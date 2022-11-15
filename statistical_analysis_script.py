import pandas as pd

solver = "Prioritized"

df = pd.read_csv(f'results_{solver}.csv')

print(df)