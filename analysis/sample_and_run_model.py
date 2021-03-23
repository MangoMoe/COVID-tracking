# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pprint import pprint
import pickle

# %%
final_means = pickle.load(open("means.pickle", "rb"))
final_cov = pickle.load(open("covariances.pickle", "rb"))
uni_cols = pickle.load(open("unique_columns.pickle", "rb"))

print(final_means.shape)
print(final_cov.shape)

# %%
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


# %%
# Got the following from this tutorial: https://goddoe.github.io/r/machine%20learning/2017/12/17/how-to-use-r-model-in-python.html
#   plus some help with environments from SO
# rpy2_wrapper/model.py 

os.environ['R_HOME'] = r"C:\Users\Dallin Clayton\.conda\envs\data\Lib\R"

import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

# manually importing here because the dependencies thing didn't work

# packages = ["ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime","rlist"]
# packages.extend( [ "zoo", "dtw", "foreach", "rlist", "data.table", "plyr", ])
print("Importing")
importr("grf")
# for pack in tqdm(packages):
#     importr(pack)

r = robjects.r
numpy2ri.activate()

class Model(object):
    """
    R Model Loader

    Attributes
    ----------
    model : R object
    """

    def __init__(self):
        self.model = None

    def load(self, path):
        model_rds_path = "{}.rds".format(path)
        model_dep_path = "{}.dep".format(path)

        print("Reading RDS")
        self.model = r.readRDS(model_rds_path)

        # print("Getting those wierd dep things")
        # with open(model_dep_path, "rt") as f:
        #     model_dep_list = [importr(dep.strip())
        #                       for dep in f.readlines()
        #                       if dep.strip()!='']

        print("Returning")
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        pred_probs : array, shape (n_samples, probs)
        """

        if self.model is None:
            raise Exception("There is no Model")
        
        if type(X) is not np.ndarray:
            X = np.array(X)

        if len(X.shape) == 1:
            X = np.reshape(X, (1, X.shape[0]))
        rows, cols = X.shape
        print("Rows: {}\nCols: {}".format(rows, cols))
        X_mat = r.matrix(X, nrow=rows, ncol=cols)

        # pred = r.predict(self.model, X, probability=True)
        # pred = r.predict(self.model, X)
        pred = r.predict(self.model, X_mat)
        # probs = r.attr(pred, "probabilities")

        # return np.array(probs)
        return np.array(pred)

# %%
print()
print()
print("Trying to load and run the model")
model = Model()
model.load(os.path.join("..","src","model_130"))

# %%
data = create_sample()
out = model.predict(data.to_numpy())

print("Final result")
print(out.shape)
print(out)

# %%
# Now to try cotter's method

# First we need to get high and low values
var = np.diag(final_cov)
# TODO is this in the same order as means?
    # Wait its the same order, as long as we add it before re-indexing, we should be just fine
std_dev = np.sqrt(var)
high = final_means.copy()
low = final_means.copy()

# Lets try 2 std dev
high = high + 2 * std_dev
low = low - 2 * std_dev

# categorical vars
#   in the interest of time, I am going to assume all categorical variables have a max value equal to the number of unique values.
#   That doesn't hold for everything, but for most of the binary and some of the other variables it should be fine.
high_cat = uni_cols[uni_cols < threshold].copy()
low_cat = uni_cols[uni_cols < threshold].copy()
low_cat[low_cat.index] = 0
high = high.append(high_cat)
low = low.append(low_cat)

# Drop unused columns
high_cols = set(high.index)
high = high.drop(removed_cols.intersection(high_cols))
low = low.drop(removed_cols.intersection(high_cols))

# Then create a matrix of different combinations of high and low for each thing
data = create_sample()

# All high
data_high = data.copy()
data_high[high.index] = high

# Create all low
data_low = data.copy()
data_low[low.index] = low

# All low but one
print(high.shape[0])
data_mat = data_high.copy()
# TODO check this is right...
for i in range(data_low.shape[0]):
    temp = data_low.copy()
    # Includes the ones we are not changing but the high and low values are the same for those so it doesn't really matter that much
    temp.iloc[i] = data_high.iloc[i]
    data_mat = pd.concat([data_mat, temp], axis=1)

# All high but one
for i in range(data_high.shape[0]):
    temp = data_high.copy()
    temp.iloc[i] = data_low.iloc[i]
    data_mat = pd.concat([data_mat, temp], axis=1)

# Append all low
data_mat = pd.concat([data_mat, data_low], axis=1)
print(data_mat.shape)
data_mat = data_mat.T
print(data_mat.shape)

# Run the matrix through the model and record all of the results
#   You might want to split the results into 4 groups. All low, the ones with one high, the ones with one low, and all high for ease of the next computations
# res = model.predict(data_mat.to_numpy())
# print(res.shape)
res_mat = np.zeros(data_mat.shape[0])
for i in tqdm(range(data_mat.shape[0])):
    res = model.predict(data_mat.iloc[i].to_numpy())
    res_mat[i] = res[0][0]
print(res_mat.shape)

# %%
# TODO so, basically I'm not exactly sure if the model expects normalized data or not, it seems not but I can't tell.
    # Well having too big of data would be better to see this effect so we'll stay how we are doing it
# finally calculate C_o and C_e for each factor and then calculate the ordering
n = data_mat.shape[1]
first = res_mat[0]
last = res_mat[-1]
low_end = res_mat[1:n + 1]
high_end = res_mat[n+1:2*n + 1]
print(first)
print(last)
print(low_end.shape)
print(high_end.shape)

odd_effects = 0.25 * ((last - high_end) + (low_end - first))
even_effects = 0.25 * ((last - high_end) - (low_end - first))

measure = np.abs(odd_effects) + np.abs(even_effects)
print(measure.shape)

measure = pd.Series(measure, index=data.index)

# %%

# This already takes variable scale into account since that was caputred in the magnitudes of the high and low values

measure = measure.sort_values(ascending=False)
print(measure)
# index, sorter = measure.index.sort_values(ascending=False, return_indexer=True)
# print(sorter)
# print()
# measure = measure.reindex(index)
# print(measure[sorter])
# pprint(list(index))

# %%
import pickle
pickle.dump(measure,  open("cotter_measure.pickle", "wb"))
pickle.dump(res_mat,  open("cotter_2np2.pickle", "wb"))

# %%
import pickle
measure = pickle.load(open("cotter_measure.pickle", "rb"))
res_mat = pickle.load(open("cotter_2np2.pickle", "rb"))

# %%
# Lets try getting like the top 30 or something
amount = 30
print(measure.iloc[:amount])
selection = measure.iloc[:amount]