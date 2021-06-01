# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
# from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
# import stuff from scipy
import os
# print(os.getcwd())
from pprint import pprint
from functools import reduce

# %% [markdown]
'''
So we need to try to incrementally calculate the covariance.

$$
\sigma_{x,y}^2 = \frac{1}{n}\left[ \sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y}) \right] \\
\sigma_{x,y}^2 = \frac{1}{n}\left[ \sum_{i=1}^{n}(x_iy_i - \bar{x}y_i - \bar{y}x_i + \bar{x}\bar{y}) \right] \\
\sigma_{x,y}^2 = \frac{1}{n}\left[ \sum_{i=1}^{n}(x_iy_i) \right] - \frac{1}{n}\left[ \sum_{i=1}^{n}(\bar{x}y_i) \right] - \frac{1}{n}\left[ \sum_{i=1}^{n}(\bar{y}x_i) \right] + \frac{1}{n}\left[ \sum_{i=1}^{n}(\bar{x}\bar{y}) \right] \\
\sigma_{x,y}^2 = \frac{1}{n}\left[ \sum_{i=1}^{n}(x_iy_i) \right] - \frac{\bar{x}}{n}\left[ \sum_{i=1}^{n}(y_i) \right] - \frac{\bar{y}}{n}\left[ \sum_{i=1}^{n}(x_i) \right] + \bar{x}\bar{y} \\
\sigma_{x,y}^2 = \frac{1}{n}\left[ \sum_{i=1}^{n}(x_iy_i) \right] - \bar{x}\bar{y} \\
$$

(Actually we just ended up subsampling and getting the covariance from that...)
'''

# %%
vec = np.array([[1,2,3],[4,5,6]])
# mat = np.outer(vec,vec)
mat = np.outer(vec[0], vec[0]) + np.outer(vec[1], vec[1])
print(mat)
mat = vec.T@vec
print(mat)

# %%
def check_csv(csv_file, columns=None):
    df = pd.read_csv(csv_file)
    df = df.select_dtypes(include=np.number)
    # NOTE: we are going to only calculate distributions over non-categorical variables,
    #   we will still have to account for them in our analysis, but we will use something like the morris method and set the categorical ones to each possible value.
    cols = df.columns
    num_unique = []
    for col in cols:
        num_unique.append(len(np.unique(df[col])))
    threshold = 10
    cols = np.array(cols)
    num_unique = np.array(num_unique)
    print("Size before")
    print(df.shape)
    df = df[cols[num_unique > threshold]]
    print("Size after")
    print(df.shape)

    if columns is not None:
        print()
        print("Printing dataframe's supposedly negative columns")
        print(df[columns])


    # cols = list(df.columns)
    # pprint(cols)
    # print(len(cols))
    # print(df.head())
    # thing = df.apply(lambda x: x.isna().sum(), axis='rows')
    # print(thing.to_numpy().nonzero()[0].shape)
    # thing = df.apply(lambda x: x.notnull().sum(), axis='rows')
    # print(thing.to_numpy().nonzero()[0].shape)
    df = df.dropna(axis='columns', how='all')
    df = df.fillna(0)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=lambda x: x.strip())
    # pprint((df.columns))
    # print(df.columns[161])
    # print(df.iloc[:,0])
    # print(df.shape)
    # print(df.mean())
    # print(df.median())
    # print(df.var())
    # print(df.describe())
    # return df
    # mean = df.mean()
    # var = df.var()
    summy = df.dropna(how="all").sum()
    vecs = df.dropna(how="all").to_numpy()
    sum_squ = pd.DataFrame(vecs.T@vecs, index=df.columns, columns=df.columns)
    # sum_squ = (df ** 2).dropna(how="all").sum()
    n = df.count()
    ok = True
    if summy.isna().values.any() or sum_squ.isna().values.any() or n.min() <= 1:
        ok = False
        print(summy.isna().sum().sum())
        print(sum_squ.isna().sum().sum())
        print(n)
    # return summy.fillna(summy.mean()), sum_squ, n, ok, df.max()
    return summy.fillna(summy.mean()), sum_squ, n, ok

file_path = os.path.join("..", "data", "block_windowsize=2", "block_51.csv")
# summy, sum_sq, n, ok, maxie = check_csv(file_path)
summy, sum_sq, n, ok = check_csv(file_path)
mean = summy / n
print("Did we start out ok?")
print(ok)
print()
print(summy.isna().values.any())
print(sum_sq.isna().values.any())
print()

# 51-401
# for i in tqdm(range(52, 402)):
for i in tqdm(range(52, 54)):
    file_path = os.path.join("..", "data", "block_windowsize=2", "block_{}.csv".format(i))
    # summy_n, sum_sq_n, n_n, ok, max_n = check_csv(file_path)
    summy_n, sum_sq_n, n_n, ok = check_csv(file_path)
    # TODO manually confirm for small values that n is being applied correctly in the calculations
    n,  n_n = n.align(n_n, fill_value=0)
    # n_n,  n = n_n.align(n, fill_value=0)
    n = n + n_n
    # So the result of only using columns present in the current mean's index will be 
    #   that only columns common to all counties will be counted, which might be good anyway
    #   but it is good to keep in mind, especially because
    #   this is done in several places
    # mean = mean.combine((((-1 * n_n) * mean).combine(summy[mean.index], pd.Series.add)) / n, pd.Series.add).dropna()

    # maxie,  max_n = maxie.align(max_n, fill_value=0)
    # maxie = pd.concat([maxie, max_n])

    # print()
    # print(mean.isna().sum())
    # print(summy.isna().sum())
    # print('-'*10)
    mean, summy = mean.align(summy, fill_value=0)
    # print(mean.isna().sum())
    # print(summy.isna().sum())
    # print(n.isna().sum())
    # print('-'*10)
    mean = mean + (summy - n_n * mean) / n
    # print(mean.isna().sum())
    # print('-'*10)
    mean = mean.dropna()
    # print(mean.isna().sum())
    # print('-'*10)

    # print(summy.shape)
    # print(summy_n.shape)
    # print(summy.shape[0] + summy_n.shape[0])
    # print(len(set(summy.index)))
    # pprint(set(summy.index))
    # pprint(set(summy_n.index))
    # print(len(set(summy.index).union(set(summy_n.index))))
    # summy = summy.combine(summy_n, pd.Series.add).dropna()
    summy, summy_n = summy.align(summy_n, fill_value=0)
    # summy_n, summy = summy_n.align(summy, fill_value=0)
    summy = summy + summy_n
    summy = summy.dropna()
    # print(summy.shape)
    # sum_sq = sum_sq.combine(sum_sq_n, pd.Series.add).dropna()
    # sum_sq, sum_sq_n = sum_sq.align(sum_sq_n, fill_value=0)
    sum_sq_n, sum_sq = sum_sq_n.align(sum_sq, fill_value=0)
    sum_sq = sum_sq + sum_sq_n
    sum_sq = sum_sq.dropna()
    if not ok:
        print("Something went sour at index {}".format(i))
    # else:
    #     print("Everything was okay somehow???")
    #     print(summy_n.isna().values.any())
    #     print(sum_sq_n.isna().values.any())
    #     print(mean.isna().values.any())
    #     print(summy.isna().values.any())
    #     print(sum_sq.isna().values.any())

print()
print("N: {}".format(n.max()))

# print()
# print("Maximum values.")
# print(maxie)

# print()
# print("Maximum maximum value")
# print(maxie.max())

print()
print("overall means:")
# print("yeah")
# print(mean.index.duplicated(keep=False).any())
# pprint((mean.index))
print(mean.shape)
# print(mean)
print()
if mean.isna().values.any():
    print("Hey mean had problems too.")
# else:
#     print("Bro mean was fine")

# print(summy_n)
# print()
# print()
# print(sum_sq.head())
# print()
# print()
# print(summy.head())
# print()
# print()

# var = sum_sq / n - (summy / n) ** 2
# Just a reminder, the default join for align is an outer join, which is what we are looking for
mean, sum_sq = mean.align(sum_sq, fill_value=0)
means = mean.to_numpy()
# Adding tuples concatenates them
means_mat = np.zeros((means.shape + means.shape))
for i in range(len(means)):
    for j in range(len(means)):
        means_mat[i,j] = means[i] * means[j]

print("Problem dimensions")
print(sum_sq.shape)
print(n.shape)
print(means_mat.shape)
var = sum_sq / n - means_mat
print()
# TODO maybe find the covariance incrementally
print("Variance:")
print(var.shape)
# print(var)

print()
if var.isna().values.any():
    print("Variance calculation returned null...")
# else:
#     print("No NaNs in var")
covariance = var.to_numpy()

# rows, cols = covariance.shape
# for i in range(rows):
#     for j in range(cols):
#         if covariance[i,i] == 0:
#             covariance[i,i] += 10e-6
#         if covariance[j,j] == 0:
#             covariance[j,j] += 10e-6
#         if covariance[i,j] > 100000 \
#             and np.sqrt(np.abs(covariance[i,i])) * np.sqrt(np.abs(covariance[j,j])) < 100:
#             print("Well this is a problem...")
#             print("Covariance: {}".format(covariance[i,j]))
#             print("Normalizaiton factor: {}".format(np.sqrt(np.abs(covariance[i,i])) * np.sqrt(np.abs(covariance[j,j]))))
#             print("Parts of normalization factor:")
#             print(np.sqrt(np.abs(covariance[i,i])))
#             print(np.sqrt(np.abs(covariance[j,j])))
#             break
#         covariance[i,j] /= np.sqrt(np.abs(covariance[i,i])) * np.sqrt(np.abs(covariance[j,j])) 


# %%
print(sum_sq.to_numpy().max())
print(means_mat.max())
print(means.shape)
print(covariance.shape)
cov_sym = (covariance + covariance.T) / 2
print(np.all(np.linalg.eigvals(cov_sym) > 0))
print(np.min(np.linalg.eigvals(cov_sym)))
covariance = cov_sym
# print(np.linalg.eigvals(cov_sym))
# print(covariance)

# samples = np.random.multivariate_normal(mean=means, cov=covariance, size=100)
# print(samples.shape)
# print(samples[0])

# %%
# spread = 2
# for loc in range(covariance.shape[0] - spread):
#     print(covariance[loc:loc + spread,loc:loc + spread])
pprint(list(mean.index))
# TODO this is temporary
# print(mean.index)
plt.imshow(covariance, cmap='hot', interpolation="nearest")
plt.show()

# %%
covariance_threshold = 1000000

# %%
print(np.allclose(covariance, covariance.T))
vals = np.unravel_index(np.argmax(covariance), covariance.shape)
# print(vals)
# print(covariance[vals])
while covariance[vals] > covariance_threshold:
    covariance[vals] = 0.0
    vals = np.unravel_index(np.argmax(covariance), covariance.shape)
    # print(vals)
    # print(covariance[vals])
print(covariance[vals] > covariance_threshold)
plt.imshow(covariance, cmap='hot', interpolation="nearest")
plt.show()

# %%
print(np.allclose(covariance, covariance.T))
vals = np.unravel_index(np.argmin(covariance), covariance.shape)
# print(vals)
# print(covariance[vals])
while covariance[vals] < -covariance_threshold:
    covariance[vals] = 0.0
    vals = np.unravel_index(np.argmin(covariance), covariance.shape)
    # print(vals)
    # print(covariance[vals])
print(covariance[vals] > covariance_threshold)
plt.imshow(covariance, cmap='hot', interpolation="nearest")
plt.show()

# %%
print(vals)
print(covariance[vals])
print(np.max(covariance))
print(np.min(covariance))

# # %%
# ar = np.array([[1,2,3], [4,5,16], [7,8,9]])
# print(ar)
# print(np.argmax(np.max(ar, axis=0)))
# print(np.argmax(np.max(ar, axis=1)))

# %%
# Okay we are going to go with subsampling
def check_csv(csv_file, num_samples=5, columns=None):
    df = pd.read_csv(csv_file)
    df = df.select_dtypes(include=np.number)
    df = df.fillna(0)
    df.columns = df.columns.str.strip()
    df_pos = (df >= 0).T[(df >= 0).all(axis=0)].index
    if columns is not None:
        print()
        print("Produced index:")
        print(df_pos)
        print()
        print("Possibly negative columns...")
        print(df[df[columns] < 0].head())

    # NOTE: we are going to only calculate distributions over non-categorical variables,
    #   we will still have to account for them in our analysis, but we will use something like the morris method and set the categorical ones to each possible value.
    # cols = df.columns
    # num_unique = []
    # for col in cols:
    #     num_unique.append(len(np.unique(df[col])))
    # threshold = 10
    # cols = np.array(cols)
    # num_unique = np.array(num_unique)
    # # print("Size before")
    # # print(df.shape)
    # df = df[cols[num_unique > threshold]]

    # df = df[final_cols]


    # print("Size after")
    # print(df.shape)
    # sample has a lot of different modes, like percentages and whatnot, so try those out too
    # res = df.sample(num_samples)
    # print(res.shape)
    # del df
    # uni_cols = pd.Series(num_unique, index=cols)
    # return res, uni_cols, df_pos.to_series()
    return None, None, df_pos.to_series()

means = []
covs = []
indexes = []
uni_cols = None
# for i in tqdm(range(10)):
# for i in range(5):
for i in range(1):
    print("\nIteration {}".format(i + 1))
    samples, uni_cols, pos_cols = check_csv(os.path.join("..", "data", "block_windowsize=2", "block_{}.csv".format(51)))
    # print(samples.shape)
    for i in tqdm(range(52, 402)):
    # for i in tqdm(range(84, 86)):
    # for i in tqdm(range(52, 54)):
        if i == 83 or i == 84:
            continue
        file_path = os.path.join("..", "data", "block_windowsize=2", "block_{}.csv".format(i))
        # summy_n, sum_sq_n, n_n, ok, max_n = check_csv(file_path)
        # subset, uni_cols_n, pos_cols_n = check_csv(file_path, columns=pos_cols)
        subset, uni_cols_n, pos_cols_n = check_csv(file_path)
        # print(pos_cols)
        # uni_cols, uni_cols_n = uni_cols.align(uni_cols_n, fill_value=0)
        # uni_cols = pd.concat([uni_cols, uni_cols_n], axis=1).max(axis=1)
        # samples, subset = samples.align(subset, fill_value=0, axis=1)
        # samples = pd.concat([samples, subset])

        pos_cols, pos_cols_n = pos_cols.align(pos_cols_n, join="inner", fill_value=0)

    # mean = samples.mean()
    # means.append(mean)
    # cov = samples.cov()
    # covs.append(cov)
    # indexes.append(mean.index)

print()
print(pos_cols.index)
print()
print(pos_cols.shape)

# %%
from pprint import pprint
pprint(list(pos_cols.index))
print()
print('totalTestsViral' in list(pos_cols.index))

# %%
# use the mean
print()
print()
# print(means[0])
    # n,  n_n = n.align(n_n, fill_value=0)
def special_merge(left, right):
    # TODO outer align here and elsewhere?
    left, right = left.align(right, fill_value=0, axis=0) 
    return pd.concat([left, right], axis=1)
mean = reduce(lambda left, right: special_merge(left, right), means)
# means = pd.concat(means, axis=1)
print(mean.shape)
final_means = mean.mean(axis=1)
print(final_means.shape)
# print(final_means)

print()

cov = np.stack([np.nan_to_num(x.to_numpy()) for x in covs], axis=0)
print(cov.shape)
final_cov = np.mean(cov, axis=0)
print(final_cov.shape)

# %%
plt.imshow(cov_sym, cmap='hot', interpolation="nearest")
plt.show()
plt.figure()
plt.imshow(final_cov, cmap='hot', interpolation="nearest")
plt.show()

# %%
# import pickle
# pickle.dump(final_means,  open("means.pickle", "wb"))
# pickle.dump(final_cov,  open("covariances.pickle", "wb"))
# pickle.dump(uni_cols, open("unique_columns.pickle", "wb"))

# %%
import pickle
final_means = pickle.load(open("means.pickle", "rb"))
final_cov = pickle.load(open("covariances.pickle", "rb"))
uni_cols = pickle.load(open("unique_columns.pickle", "rb"))

# %%
threshold = 10
# print()
# print(np.max(cov_sym))
# print(np.min(cov_sym))
print()
print(np.max(final_cov))
print(np.min(final_cov))
print(final_cov.shape)
# print(uni_cols)
print(uni_cols.shape)
print(uni_cols[uni_cols < threshold])
print(uni_cols[uni_cols < threshold].shape)

# %%
# print(final_means.index)
# final_cols = final_means.index

# %%
samples = np.random.multivariate_normal(mean=final_means, cov=final_cov, size=100)
print(samples.shape)
# print(samples[0])

# %%
print(samples.shape[1] + uni_cols[uni_cols < threshold].shape[0])
# I think for everything I don't have I'll just input zero or something
