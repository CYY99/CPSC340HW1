def predict(X):
    if X[0] > - 80.5:
        y = 0
    else:
        if X[0] > 37.5:
            y = 0
        else:
            y = 1
    return y
