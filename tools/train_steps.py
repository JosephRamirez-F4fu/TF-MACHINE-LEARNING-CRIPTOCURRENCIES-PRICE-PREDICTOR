import numpy as np
def create_dataset(X, y, time_steps=1,n_future=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - n_future):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps:i + time_steps + n_future])
    return np.array(Xs), np.array(ys)