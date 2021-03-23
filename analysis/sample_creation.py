import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pprint import pprint
import pickle

final_means = pickle.load(open("means.pickle", "rb"))
final_cov = pickle.load(open("covariances.pickle", "rb"))
uni_cols = pickle.load(open("unique_columns.pickle", "rb"))

print(final_means.shape)
print(final_cov.shape)

threshold = 10
# print(uni_cols.shape)
# print(uni_cols[uni_cols < threshold])
# print(uni_cols[uni_cols < threshold].shape)

# %%
samples = np.random.multivariate_normal(mean=final_means, cov=final_cov, size=100)
# print(samples.shape)
# print(samples[0])

# %%
# print(samples.shape[1] + uni_cols[uni_cols < threshold].shape[0])
# I think for everything I don't have I'll just input zero or something

# %%
# file_path = os.path.join("..", "data", "block_windowsize=2", "block_51.csv")
file_path = os.path.join("..", "data", "block_windowsize=2", "block_400.csv")
df = pd.read_csv(file_path)
# print(df.shape)
count = 0
other_cols = []
for column in df.columns:
    if column not in final_means.index and column not in uni_cols.index:
        count += 1
        # print(column)
        other_cols.append(column)
# print("Number of other columns: {}".format(count))
# print("Total we have")
# print(samples.shape[1] + uni_cols[uni_cols < threshold].shape[0] + count)
# print("Total we need: 348")

# %%
df_cols = set(list(df.columns))
listo = []
listo.extend(list(final_means.index))
listo.extend(list(uni_cols.index))
listo.extend(other_cols)
our_cols = set(listo)
# pprint(df_cols)
# pprint(our_cols)
# print(len(df_cols))
# print(len(our_cols))
# print("\nTHE FINAL THINGY")
removed_cols = {"shifted_log_rolled_cases","datetime","State_FIPS_Code","county","state","log_rolled_cases.x","shifted_time"}
# pprint(df_cols - our_cols)
# print(len(df_cols.intersection(our_cols)))
# print(len(df_cols - removed_cols))

# %%
# Okay so it seems like we have the right amount of things now, just gotta fill those in
def create_sample():
    sample = np.random.multivariate_normal(mean=final_means, cov=final_cov, size=1)
    # print(sample[0].shape)
    # print(final_means.index.shape)
    data = pd.Series(sample[0], index=final_means.index)
    # TODO randomly pick ones or 0s for some of these
    other_data = [0] * len(uni_cols[uni_cols < threshold])
    new_index = uni_cols[uni_cols < threshold].index
    other_data = pd.Series(other_data, index=new_index)
    data = data.append(other_data)
    # These might be string, but the model only takes numeric values, so...
    for col in other_cols:
        data[col] = 0

    # print(data.shape)
    # print(len(df_cols))
    excluded = df_cols - set(data.index)
    # print(excluded)

    extra = pd.Series(0, excluded)
    data = data.append(extra)
    # data[list(excluded)] = 0
    # for thing in excluded:
    #     data[thing] = 0

    data = data.reindex(df.columns)
    data = data.drop(removed_cols)
    # print(data.shape)
    return data