from collections import Counter
import pandas as pd
import numpy as np 
import math


class Node:

    def __init__(self , feature=None , threshold=None , leaf=None , max_label=None):
        self.feature = feature

        self.threshold = threshold

        self.child = dict()

        self.leaf = leaf

        self.max_label = max_label


def entropy(y):

    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def majority_label(y):
    return y.value_counts().idxmax()

def categorical_ig(x , y , column):
    base_ent = entropy(y)

    values , counts = np.unique(x[column] , return_counts=True)


    weighted_ent = 0.0

    for value , count in zip(values,counts):
        sub_y =   y[x[column] == value]
        weighted_ent += (count / len(y)) * entropy(sub_y)

    return base_ent - weighted_ent

def best_numeric_feature(X , y , column):

    x = X[column].sort_values()

    threshold = [(x.iloc[i] + x.iloc[i+1]) / 2 for i in range(len(x) - 1)]

    best_thresh = None
    best_gain = -1
    base_ent = entropy(y)

    size = len(y)
    
    for thresh in threshold:
        left = y[x <= thresh]
        right = y[x > thresh]

        if len(left) == 0 or len(right) == 0:
            continue

        gain = base_ent - ( (len(left) / size) * entropy(left) + (len(right) / size) * entropy(right))

        if gain > best_gain:
            best_gain = gain
            best_thresh = thresh
    
    return best_gain , best_thresh


def choose_best_feature(x,y):

    best_feature = None
    best_threshold = None
    best_type = None
    best_gain  = -1

    for feat in x.columns:

        if x[feat].dtype == "object":
            gain = categorical_ig(x , y , feat)
            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                best_type = "categorical"
                best_threshold = None
        
        else:
            gain , thresh= best_numeric_feature(x , y , feat)

            if gain > best_gain:
                best_gain = gain
                best_feature = feat
                best_type = "numerical"
                best_threshold = thresh

    return best_feature , best_type , best_threshold


def build_tree(x, y, depth=0, max_depth=3):

    if len(set(y)) == 1:
        return Node(leaf = y.unique()[0])

    if len(x.columns) == 0:
        label = majority_label(y)

        return Node(leaf=label , max_label=label)
    
    
    if depth >= max_depth:
        label = majority_label(y)
        return Node(leaf=label , max_label=label)


    feature , ftype , threshold = choose_best_feature(x,y)
    
    if feature not in x.columns:
        
        label = majority_label(y)

        return Node(leaf=label , max_label=label)


    root = Node(feature=feature , threshold=threshold , max_label=majority_label(y))

    if ftype == "categorical":

        for value  in x[feature].unique():

            sub_x = x[x[feature] == value].drop(columns=[feature]) 

            sub_y = y.loc[sub_x.index]

            child = build_tree(sub_x,sub_y,depth+1 , max_depth)

            root.child[value] = child

    else:
        left_x = x[x[feature] <= threshold].drop(columns = [feature])
        left_y = y.loc[left_x.index]

        right_x = x[x[feature] > threshold].drop(columns = [feature])
        right_y = y.loc[right_x.index]

        if len(left_y) == 0 or len(right_y) == 0:
            return Node(leaf=majority_label(y) , max_label=majority_label(y))

        root.child["left"] = build_tree(left_x , left_y, depth +1 , max_depth)
        root.child["right"] = build_tree(right_x , right_y, depth+1 , max_depth)
    
    return root

def print_tree(node):

    if node.leaf is not None:
        print("----> leaf ",node.leaf)
        return

    print("feature: ",node.feature)

    for value , child in node.child.items():

        print(f"---> {value}")
    
        print_tree(child)

def predict_one(node,test):

    if node.leaf is not None:
        
        return node.leaf
    
    feature = node.feature

    if node.threshold is not None:
        if test[feature] <= node.threshold:
            return predict_one(node.child["left"],test)
        else:
            return predict_one(node.child["right"],test)
 
    else:
        value = test[feature]

        if value in node.child:
            return predict_one(node.child[value],test)

        else:
            return node.max_label

def predict(tree,x):

    return x.apply(lambda row:predict_one(tree,row),axis=1)

def score(y_test , y_pred):

    return (y_test == y_pred).mean()

