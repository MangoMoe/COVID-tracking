# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pprint import pprint
import pickle
from sample_creation import create_sample
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

# %%
# Lets try getting like the top 30 or something
amount = 50
measure = measure.sort_values(ascending=False)
print(measure.iloc[:amount])
selection = measure.iloc[:amount]

# %%
# Now to get some samples to calculate sensitivity for
# Well first lets make *A* sample and getting the sensitivity for it

# TODO TODO I suspect that the reason we are getting zero sensitivities 
#   (still not sure why this persists when delta is high)
#   is because at this point either 
#   a) the less important factors are being used to fine-tune the decision
#   or b) multiple factors have to change
#   For a, we can try finding the local sensitivities for ALL variables, 
#       not just the top 30
#   For b, it might be better to use monte carlo sampling (maybe in a small area?) 
#       and try to use regression to estimate the sensitivities, 
#       since we know there are differences in the values when we monte carlo sample

# point should be a pandas object
def sensitivity_analysis(model, point, selection, means, std_devs, delta=0.50):
    sensitivity = dict()
    q_0 = model.predict(point.to_numpy())
    print(q_0)
    for column in tqdm(selection.index):
        # This is probably inefficient...
        shift_point = point.copy()
        # TODO adding standard deviations to try to get something to happen...
        shift_point[column] += delta * std_devs[column]
        print("------")
        print()
        print((shift_point - point)[column])
        print(std_devs[column])
        q = model.predict(shift_point.to_numpy())
        print()
        print(q)
        sensitivity[column] = (q - q_0) / delta
    
    sensitivity = pd.Series(sensitivity)
    return sensitivity

point = create_sample()
sens = sensitivity_analysis(model, point, selection, final_means, std_devs, delta=10.0)
sens = sens.sort_values(ascending=False)
print()
print(sens)

# %%
point = create_sample()
q = model.predict(point.to_numpy())
print(q)
