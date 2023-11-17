import random
import sys
import time

import numpy as np
import pandas as pd

min_samples = range(2, 21)
depth = range(2, 11)
mode = ['entropy', 'gini']


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2, mode='entropy'):
        ''' constructor '''

        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if children are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, self.mode)
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''

        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        ''' function to compute entropy '''

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        ''' function to compute gini index '''

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print(col_names[tree.feature_index + 1], "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % indent, end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % indent, end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        ''' function to train the tree '''

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' function to predict new dataset '''

        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''

        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


def train_test_split(X, Y, test_size, seed):
    if test_size == 0.0:
        return X, [], Y, []
    elif test_size == 1.0:
        return [], X, [], Y
    dataset = np.concatenate((X, Y), axis=1)
    train = np.zeros(shape=(len(X) - int(test_size * len(X)), len(col_names)))
    test = np.zeros(shape=(int(test_size * len(X)), len(col_names)))
    indices = list(range(len(X)))
    if seed is None:
        seed = random.randrange(sys.maxsize)
        # print("Seed was: " + str(seed))
    random.seed(seed)
    n = int(test_size * len(X))
    for i in range(n):
        index = random.randint(0, len(indices) - 1)
        test[i] = dataset[indices[index]]
        indices.pop(index)
    n = len(X) - n
    for i in range(n):
        index = random.randint(0, len(indices) - 1)
        train[i] = dataset[indices[index]]
        indices.pop(index)
    X_train, Y_train = train[:, :-1], train[:, -1].reshape(-1, 1)
    X_test, Y_test = test[:, :-1], test[:, -1].reshape(-1, 1)
    return X_train, X_test, Y_train, Y_test


def accuracy_score(test, pred):
    count = 0
    for i in range(len(test)):
        if test[i] == pred[i]:
            count = count + 1
    return count / len(test)


def random_search(X, Y, iter):
    best_s = 0
    best_d = 0
    best_m = 0
    best_score = 0.0
    random.seed(time.time())
    for i in range(iter):
        s = min_samples[random.randint(0, len(min_samples) - 1)]
        d = depth[random.randint(0, len(depth) - 1)]
        m = random.randint(0, 1)
        classifier = DecisionTreeClassifier(min_samples_split=s, max_depth=d, mode=mode[m])
        classifier.fit(X, Y)
        Y_pred = classifier.predict(X)
        score = accuracy_score(Y, Y_pred)
        if score > best_score:
            best_s = s
            best_d = d
            best_m = m
            best_score = score
        if best_score == 1:
            break
    return best_s, best_d, best_m


col_names = ['type', 'alcohol', 'malic_acid', 'ash', 'alkalinity', 'magnesium', 'phenols', 'flavanoids', 'nonflavanoid',
             'proanthocyanins', 'color_intensity', 'hue', 'diluted_wines', 'proline']
data = pd.read_csv("wines.csv", header=None, names=col_names)

# start = time.time()
X = data.iloc[:, 1:].values
Y = data.iloc[:, 0].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, .3, None)
X_train2, X_valid, Y_train2, Y_valid = train_test_split(X_train, Y_train, .3, None)

s, d, m = random_search(X_valid, Y_valid, 15)
print("s = " + str(s) + ", d = " + str(d) + ", m = " + mode[m])
classifier = DecisionTreeClassifier(min_samples_split=s, max_depth=d, mode=mode[m])
classifier.fit(X_train, Y_train)
classifier.print_tree()

Y_pred = classifier.predict(X_test)
print("Accuracy: " + str(accuracy_score(Y_test, Y_pred)))
# end = time.time()
# print(end - start)