# %%
import os
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

        # TODO this might cause problems but probably won't
        # return np.array(probs)
        return np.array(pred)[0][0]

# %%
# print()
# print()
# print("Trying to load and run the model")
# model = Model()
# model.load(os.path.join("..","src","model_130"))

# # %%
# data = create_sample()
# out = model.predict(data.to_numpy())