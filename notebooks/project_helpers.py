from time import time
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import Adam
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator

#Credit: https://www.geeksforgeeks.org/timing-functions-with-decorators-python/
def time_this(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

# Full credit to scikeras tutorial (see 2.2 in 'tutorial' link above)
def get_clf(meta, hidden_layer_sizes, dropout):
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta["n_classes_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation="relu"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    return model

# DummyEstimator to act as proxy for clf's in Random Search vs
class DummyEstimator(BaseEstimator):
    def fit(self): pass
    def score(self): pass