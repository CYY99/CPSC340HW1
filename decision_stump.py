import numpy as np
import utils


#     The ﬁle decision_stump.py contains the class DecisionStumpEquality
#     which ﬁnds the best decision stump using the equality rule and
#     then makes predictions using that rule.
#
#     Instead of discretizing
#     the data and using a rule based on testing an equality for a single feature,
#     we want to check whether a feature is above or below a threshold and
#     split the data accordingly (this is a more sane approach, which we discussed in class).
#     Create a DecisionStumpErrorRate class to do this, and report the updated error you obtain
#     by using inequalities instead of discretizing and testing equality.
#     Also submit the generated ﬁgure of the classiﬁcation boundary.
#     Hint: you may want to start by copy/pasting the contents DecisionStumpEquality
#     and then make modiﬁcations from there.

class DecisionStumpEquality:

    def __init__(self):

        pass

    def fit(self, X, y):
        N, D = X.shape  # N is the num of examples , D is num of features

        # here X is a 400 * 2 matrix
        # here Y is a 400 * 1 matrix with value of 0 and 1

        # print(N) here N = 400
        # print(D) here D = 2

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)
        # here count = [234 166], num of 0 = 234, num of 1 = 166
        # print(count)

        # Get the index of the largest label value in count.
        # which means y_mode = index(0 or 1) of which label value(0 or 1) has more
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)
        # y_mode is most popular label value, here is 0

        # print(y_mode)

        self.splitSat = y_mode  # = 0 ?????
        self.splitNot = None  # ?????
        self.splitVariable = None  # split feature for Equality
        self.splitValue = None  # ?????

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)
        # here minError = 166
        # print(y)
        # print(y != y_mode)
        # print((minError))

        # Loop over features looking for the best split
        X = np.round(X)
        # round each Xnd of X to int

        # value = X[1, 1]

        # print(value) # 32
        # print(X[:, 1])
        # y_sat = utils.mode(y[X[:, 1] == value])
        # print(y_sat) # 1
        # print(y[X[:, 1] == value]) # [1 1 1 1 1 1 0 1 1 1 1 1]
        # print(y[X[:, 1] != value])
        # print(utils.mode(y[X[:, 1] != value]))
        # Find most likely class for each split
        # First, let's look at the X[:,d] == value. This is a condition,
        # it means that the d column of X which has the same value with "value" should be true.
        # then y[condition] gives you labels for the rows which are true
        # (labels of the points which satisfy the equality rule in decision stump).

        for d in range(D):  # outer loop each feature
            for n in range(N):  # inner loop each example
                # Choose value to equate to
                value = X[n, d]  # Xij, or we say Xnd, as our current equality value

                # Find most likely class for each split
                # "y[X[:,d] == value" gives the values in y where the corresponding Xnd = the current equality value
                # it returns a part of y
                # y_sat is the most appeared label value in y[X[:,d] == value]
                # y_not is the most appeared label value in y[X[:,d] != value]
                y_sat = utils.mode(y[X[:, d] == value])
                y_not = utils.mode(y[X[:, d] != value])

                # Make predictions
                y_pred = y_sat * np.ones(N)  # = [1 1 1 ...] or [0 0 0 ...]
                y_pred[X[:, d] != value] = y_not  # change those y[X[:,d] != value] corresponding y[] to y_not

                # Compute error
                errors = np.sum(y_pred != y)
                # y_pred != y will give a array where not equivalent y_pred and y value will be 1

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d  # change the split feature to the new feature
                    self.splitValue = value  # change the equality value to the new value
                    self.splitSat = y_sat  # predicted y if Xnd == splitValue
                    self.splitNot = y_not  # predicted y if Xnd != splitValue

    def predict(self, X):

        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] == self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat


class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape  # N is the num of examples , D is num of features

        # here X is a 400 * 2 matrix
        # here Y is a 400 * 1 matrix with value of 0 and 1

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)
        # here count = [234 166], num of 0 = 234, num of 1 = 166

        # Get the index of the largest label value in count.
        # which means y_mode = index(0 or 1) of which label value(0 or 1) has more
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)
        # y_mode is most popular label value, here is 0

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None  # split feature for Equality
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)
        # here minError = 166
        # print(y)
        # print(y != y_mode)
        # print((minError))

        # Loop over features looking for the best split
        # X = np.round(X)
        # round each Xnd of X to int

        # value = X[1, 1]

        # print(value) # 32
        # print(X[:, 1])
        # y_sat = utils.mode(y[X[:, 1] == value])
        # print(y_sat) # 1
        # print(y[X[:, 1] == value]) # [1 1 1 1 1 1 0 1 1 1 1 1]
        # print(y[X[:, 1] != value])
        # print(utils.mode(y[X[:, 1] != value]))
        # Find most likely class for each split
        # First, let's look at the X[:,d] == value. This is a condition,
        # it means that the d column of X which has the same value with "value" should be true.
        # then y[condition] gives you labels for the rows which are true
        # (labels of the points which satisfy the equality rule in decision stump).

        for d in range(D):  # outer loop each feature
            for n in range(N):  # inner loop each example
                # Choose value to equate to
                value = X[n, d]  # Xij, or we say Xnd, as our current inequality value

                # Find most likely class for each split
                # "y[X[:,d] > value" gives the values in y where the corresponding Xnd > the current equality value
                # it returns a part of y
                # y_sat is the most appeared label value in y[X[:,d] > value]
                # y_not is the most appeared label value in y[X[:,d] <= value]
                y_sat = utils.mode(y[X[:, d] > value])
                y_not = utils.mode(y[X[:, d] <= value])

                # Make predictions
                y_pred = y_sat * np.ones(N)  # = [1 1 1 ...] or [0 0 0 ...]
                y_pred[X[:, d] <= value] = y_not  # change those y[X[:,d] <= value] corresponding y[] to y_not

                # Compute error
                errors = np.sum(y_pred != y)
                # y_pred != y will give a array where not equivalent y_pred and y value will be 1

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d  # change the split(inequality) feature to the new feature
                    self.splitValue = value  # change the spilt(inequality) value to the new value
                    self.splitSat = y_sat  # predicted y if Xnd > splitValue
                    self.splitNot = y_not  # predicted y if Xnd <= splitValue

    def predict(self, X):

        M, D = X.shape
        # X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] > self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat


"""
A helper function that computes the entropy of the 
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to 1.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true (you can show this with l'Hopital's rule), 
but numerically results in NaN because log(0) returns -Inf.
"""


def entropy(p):
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)


# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?
class DecisionStumpInfoGain(DecisionStumpErrorRate):

    def fit(self, X, y):
        # set the split_features to default Nore if its value not given in the parameter
        # here split_features is the
        N, D = X.shape  # N is the num of examples , D is num of features

        # here X is a 400 * 2 matrix
        # here Y is a 400 * 1 matrix with value of 0 and 1

        # Entropy Function PseudoCode
        # ----------------------------------------------
        # vector 'y' of length 'n' with numbers {1,2,3, ...k}
        # counts = zeros(k)
        # for i in 1:n
        #     counts[y[i]] += 1
        # entropy = 0
        # for c in 1:k
        #     prob = counts[c]/n
        #     entropy -= prob * log(prob)
        # return entropy
        # -----------------------------------------------
        # -------------------equivalent----------------------
        # for i in 1:n
        #     counts[y[i]] += 1
        # entropy = 0
        # ---------------------equivalent--------------------
        count = np.bincount(y)
        # in this example count = [234 166], num of 0 = 234, num of 1 = 166
        # length = amax + 1, here = 2
        # index 0 has 234, which is num of 0
        # index 1 has 166, which is num of 1

        countLength = len(count)
        # this will be used later
        # here countLength = 1 + max(y)

        prob = count/np.sum(count)  # here ym = n in the pseudo code, which is the number of rows of y
        etp_before_split = entropy(prob)

        # Get the index of the largest label value in count.
        # which means y_mode = index(0 or 1) of which label value(0 or 1) has more
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)
        # y_mode is most popular label value, here is 0

        max_info_gain = 0  # Information gain for baseline rule (“do nothing”) is 0.
        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        # X = np.round(X)
        # round each Xnd of X to int

        for d in range(D):  # outer loop each feature
            feature_d = np.unique(X[:,d])
            for value in feature_d[:-1]:

                count_yes = np.bincount(y[X[:, d] > value], minlength=countLength)
                count_not = np.bincount(y[X[:, d] <= value], minlength=countLength)

                # info gain formula :
                # entropy(y) - n_yes/n * entropy(y_yes) - n_no/n * entropy(y_no)

                p_yes = count_yes/np.sum(count_yes)
                etp_yes = entropy(p_yes)

                p_not = count_not/np.sum(count_not)
                etp_not = entropy(p_not)

                prob1 = np.sum(X[:, d] > value) / N
                # n_yes / n

                prob0 = np.sum(X[:, d] <= value) / N
                # n_no / n

                info_gain = etp_before_split - prob1 * etp_yes - prob0 * etp_not

                # compare to max info gain
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = np.argmax(count_yes)
                    self.splitNot = np.argmax(count_not)

