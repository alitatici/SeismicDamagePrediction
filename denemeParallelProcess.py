import pandas as pd
from joblib import Parallel, delayed
import numpy as np

# create a sample DataFrame
df = pd.DataFrame({
    'A': range(1, 11),
    'B': range(11, 21),
    'C': range(21, 31)
})

# split the DataFrame into 8 equal parts
dfs = np.array_split(df, 8)

# # print the number of rows in each split DataFrame
# for i, split_df in enumerate(dfs):
#     print(f'Split {i+1} has {len(split_df)} rows')

# # create a sample DataFrame
# df = pd.DataFrame({
#     'A': [1, 2, 3, 4, 5],
#     'B': [6, 7, 8, 9, 10]
# })

# define a function to process each row
def process_row(row):
    row['D'] = row['A'] + row['B'] + row['C']
    return row

# use joblib to process each row in parallel
# results = Parallel(n_jobs=-1)(
#     delayed(process_row)(row) for _, row in df.iterrows()
# )
results = [process_row(row) for _, row in df.iterrows()]
# convert the results back to a DataFrame
df = pd.DataFrame(results)

# print the updated DataFrame
print(df)