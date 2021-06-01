# %%
os.environ['R_HOME'] = r"C:\Users\Dallin Clayton\.conda\envs\data\Lib\R"

# %%
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pprint import pprint
import pickle
from sample_creation import create_sample, fix_sample
from sklearn.linear_model import LinearRegression
from pyDOE import *
from r_model_load import Model

# %%
measure = pickle.load(open("cotter_measure.pickle", "rb"))
# res_mat = pickle.load(open("cotter_2np2.pickle", "rb"))
final_means = pickle.load(open("means.pickle", "rb"))
final_cov = pickle.load(open("covariances.pickle", "rb"))
uni_cols = pickle.load(open("unique_columns.pickle", "rb"))
var = np.diag(final_cov)
std_dev = np.sqrt(var)
std_devs = pd.Series(std_dev, index=final_means.index)

model = Model()
model.load(os.path.join("..","src","model_130"))

amount = 30
measure = measure.sort_values(ascending=False)
# print(measure.iloc[:amount])
selection = measure.iloc[:amount]
# selection = measure.iloc[:]
selection = selection.sort_values(ascending=False)
pprint(selection)

# %%
num_points = 1
std_devs_scaling = 200.0
# TODO figure out what the heck this should be
delta = 100
# TODO make this big once we know its working
num_deltas = 50

example_point = create_sample()
# removed_cols = {"shifted_log_rolled_cases","State_FIPS_Code","log_rolled_cases.x","shifted_time"}
samples = lhs(n=measure.shape[0], samples=num_points)
threshold = 10

other_data = [0] * len(uni_cols[uni_cols < threshold])
new_index = uni_cols[uni_cols < threshold].index
other_data = pd.Series(other_data, index=new_index)

# TODO remember that you are just assuming everything (means and std dev) for stuff that was categorical is just 0
means_full = final_means.append(other_data)
std_devs_full = std_devs.append(other_data)
means_full, _ = means_full.align(example_point, join='right', fill_value=0)
std_devs_full, _ = std_devs_full.align(example_point, join='right', fill_value=0)
print(means_full.shape)
print(std_devs_full.shape)
print(samples.shape)

# center them at 0
samples -= 0.5

# Scale to std_devs, multiply by 2 since we have 0.5 on either side
samples *= std_devs_full.to_numpy() * 2 * std_devs_scaling

# Add mean
samples += means_full.to_numpy()

samples_df = pd.DataFrame(samples.T, index=means_full.index).T

orig_points = []
orig_q_s = []
delta_points = []
delta_q_s = []
sample_delta_points = []
sample_delta_q_s = []
sensitivities = []

print(samples_df.T.shape)
temp_point = create_sample()

# for index, point in tqdm(samples_df.iterrows(), total=num_points):
for index, point in [(0, temp_point)]:
    # Choose a nominal point
    # p_nom = create_sample()
    # print("\n\nYooorp")
    # print(point.shape)
    p_nom = point
    orig_points.append(p_nom.copy())
    q_nom = model.predict(p_nom.to_numpy())
    orig_q_s.append(q_nom)

    # Sample points nearby, TODO how many?
    #   probably with a uniform distribution scaled by the standard deviations of each thing, 
    #   and scaled with a delta to be within a certain distance from the original point.
    points = []
    sample_points = []
    q_s = []
    sample_q_s = []
    new_std_dev = std_devs.copy()
    p_nom, new_std_dev = p_nom.align(new_std_dev, join='left', fill_value=0)
    # print()
    # print("New nominal point {}".format(index))
    for i in tqdm(range(num_deltas)):
        new_point_sample = create_sample()

        new_point = p_nom.copy().to_numpy()
        unif = np.random.uniform(size=new_point.shape)
        unif -= 0.5
        unif *= 2
        shift = unif * new_std_dev * delta
        new_point += shift

        fix_sample(new_point)

        points.append(new_point)
        sample_points.append(new_point_sample)
        # Run all the points through the model.
        q_s.append(model.predict(new_point))
        sample_q_s.append(model.predict(new_point_sample))

    delta_points.append(points)
    sample_delta_points.append(sample_points)
    delta_q_s.append(q_s)
    sample_delta_q_s.append(sample_q_s)

    # TODO double check that you are doing this right...
    # Create the matrix of differences in points and a vector of the differences in outputs.
    A = np.array(points)
    q_s = np.array(q_s)
    y = q_s - q_nom

    # Use least squares regressions to get the sensitivities
    #   TODO should we be using a regularization method like lasso regression?
    # model = LinearRegression()
    # model.fit(A, y)

    sens = np.linalg.pinv(A.T@A)@A.T@y

    print(sens)

    # TODO I really hope these are in the right order...
    sens = pd.Series(sens, p_nom.index)

    sensitivities.append(sens)

# %%
# Make sure the sensitivities are nonzero
sens = sens.sort_values(ascending=False)
print(sens[selection.index])
# print(sens[selection.index])
print(sens.max())
print(sens.min())
# print(measure.shape)
# print(final_means.shape)
# print(std_dev.shape)
# print(std_devs.shape)

# %%
print(temp_point)
print()
print(points[0])
print()
print(std_devs.sort_values(ascending=False))

# %%
# delta_points.append(points)
# delta_q_s.append(q_s)
delta_points_df = pd.DataFrame(delta_points[0])
sample_delta_points_df = pd.DataFrame(sample_delta_points[0])

# for i in range(5):
#     column = random.choice(delta_points_df.columns)
for column in selection.index:
    print(column)

    plt.figure()
    # plt.scatter(delta_points_df['totalTestsViral'], delta_q_s[0])
    # plt.scatter(temp_point['totalTestsViral'], orig_q_s[0])
    plt.scatter(delta_points_df[column], delta_q_s[0], label="delta", s=1)
    plt.scatter(sample_delta_points_df[column], sample_delta_q_s[0], label="sample", s=1)
    plt.scatter(temp_point[column], orig_q_s[0], label="orig", s=10)
    plt.title(column[:45])
    plt.legend()
# TODO TODO TODO try looking at the most sensitive column
plt.show()

# TODO it looks like some things are becoming negative when they shouldn't be
#   you need to figure out which ones those are, and fix them...
# TODO maybe just subsample from existing data when getting samples? would make sure they are realistic values?

# %%
data = {
    "original points": orig_points, 
    "original qs": orig_q_s,
    "delta points": delta_points,
    "delta qs": delta_q_s,
    "sensitivities": sensitivities
    }

import pickle
pickle.dump(data,  open("random_testing.pickle", "wb"))

# %%

final_means = pickle.load(open("means.pickle", "rb"))
final_cov = pickle.load(open("covariances.pickle", "rb"))
selection = measure.iloc[:amount]
selection_ind = [final_means.index.get_loc(c) for c in selection.index if c in final_means]

pprint(final_means[selection.index])
print()
print(selection_ind)
print()
print(final_cov.shape)

print(type(final_cov))
temp = final_cov[selection_ind][:,selection_ind]
print()
print(temp.shape)

# %%


# plt.imshow(final_cov, cmap='hot', interpolation="nearest")
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(temp, cmap='hot', interpolation="nearest")

# We want to show all ticks...
ax.set_xticks(np.arange(len(selection.index)))
ax.set_yticks(np.arange(len(selection.index)))
# ... and label them with the respective list entries
label_max_size = 20
ax.set_xticklabels([x[:label_max_size] for x in selection.index])
ax.set_yticklabels([x[:label_max_size] for x in selection.index])

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# for i in range(len(selection.index)):
#     for j in range(len(selection.index)):
#         text = ax.text(j, i, temp[i, j],
#                        ha="center", va="center", color="w")

ax.set_title("Covariance matrix")
# fig.tight_layout()
plt.show()

# %%
vals = np.unravel_index(np.argmax(temp), temp.shape)
ind = selection_ind[vals[0]]
print(vals)
print(temp[vals])
print(selection.index[vals[0]])
print(selection.index[vals[1]])
print(selection.iloc[vals[0]])
print(selection.max())

# %%
print(temp[vals])
temp[vals] = 0.0
print(temp[vals])