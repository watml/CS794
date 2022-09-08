import numpy as np


def Perceptron_alg(X, y, w_0, b_0, delta=0, max_iter=100):
    """
    Run perceptron algorithm to find a linear separator (if any) of the dataset.

    Inputs:
        X: n by d data matrix, each row (e.g., X[i]) represents a data point (with d features), each column is a feature
        y: 1d array of labels, either +1 or -1
        w_0: initial weights
        b_0: initial bias
        delta: minimum threshold/margin/tolerance, must be non-negative, default is set to zero
        max_iter: maximum number of passes/rounds to run over the whole dataset

    Outputs:
        Solution w (weight) and b (bias), which in together gives the linear separator <w,x>+b=0

    """
    n, d = X.shape
    w = w_0
    b = b_0
    for t in range(max_iter):
        idx_list = np.arange(n)
        np.random.shuffle(idx_list)
        num_mistakes = 0
        for i in idx_list:
            if y[i] * (np.dot(w, X[i]) + b) <= delta:
                w = w + y[i] * X[i]
                b = b + y[i]
                num_mistakes += 1
        if num_mistakes == 0:
            break
    return w, b
