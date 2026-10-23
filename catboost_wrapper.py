"""scikit-learn compatible CatBoost wrapper.

CatBoostRegressor alters `cat_features` in its constructor, which breaks sklearn's
clone() (used by RandomizedSearchCV). This thin wrapper stores plain parameters and
builds the CatBoost model inside fit(). The app imports it so the pickle can load.
"""
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor


class CatBoostReg(RegressorMixin, BaseEstimator):
    def __init__(self, cat_features=None, iterations=1000, learning_rate=0.05, depth=6,
                 l2_leaf_reg=3.0, random_seed=42, thread_count=-1):
        self.cat_features = cat_features
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            cat_features=list(self.cat_features or []), iterations=self.iterations,
            learning_rate=self.learning_rate, depth=self.depth, l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed, thread_count=self.thread_count, verbose=0)
        self.model_.fit(X, y)
        self.feature_names_in_ = list(X.columns)
        return self

    def predict(self, X):
        return self.model_.predict(X[self.feature_names_in_])
