#Importing required libraries ...

import os
import sys
import numpy as np
import pandas as pd
import collections
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.externals.six import StringIO 
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error
from IPython.display import Image  
import pydotplus
import matplotlib.pyplot as plt


#conda_fix function to take care of compatibility of graphviz with ipython ...

def conda_fix(graph):
        path = os.path.join(sys.base_exec_prefix, "Library", "bin", "graphviz")
        paths = ("dot", "twopi", "neato", "circo", "fdp")
        paths = {p: os.path.join(path, "{}.exe".format(p)) for p in paths}
        graph.set_graphviz_executables(paths)

print("........................PART 1..........................")
#Loading dataset ...

bikes = pd.read_csv('bikes.csv')
train_data = bikes.loc[:]
X = train_data.loc[:, "season":"windspeed"]
y = train_data.loc[:, "count"]

#Train and test data split ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=None)

#Normalizing regression target ...

y_train = (y_train-min(y_train))/(max(y_train)-min(y_train))
y_test = (y_test-min(y_test))/(max(y_test)-min(y_test))

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#Learn regression tree without and restriction on max depth ...

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

#Extracting features list from the dataset ...

b = bikes.columns.get_values()
features = np.delete(b, 0)
features = np.delete(features, 10)

print("Features are:",features)


#Estimation of total no. of nodes and leaves in the decision tree without restriction...

estimator = regressor
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
leaves = 0
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True
        leaves = leaves + 1
print('Total No. of nodes = ', n_nodes)
print('Total No. of leaves = ', leaves)
depth_max = max(node_depth)
print("Depth of tree without any restriction on the maximum depth is", depth_max,".")


#Finding into how many different groups of days does the tree divide the data ...
weekdays = ["Monday", "Tuesday", "Wednesday", "Thrusday", "Friday", "Saturday", "Sunday"]
wd_index = 4
thresh = []
for i in range(n_nodes):
    if feature[i] == wd_index:
        thresh.append(threshold[i])

thresh = list(set(thresh))
thresh.sort()
print("Threshold values for weekdays:", thresh)

groups = []
for i in thresh:
    a = int(i)
    groups.append([0,a])
    groups.append([a+1,6])

b_set = set(tuple(x) for x in groups)
group = [ list(x) for x in b_set ]
group.sort(key = lambda x: groups.index(x) )
num = len(group)
print(group)
print("Hence, clearly the tree divides the data in totally", num,"groups of weekdays namely:")
for i in group:
    if i[0] == i[1]:
        print("[",weekdays[i[0]],"]")
    else:
        print("[",weekdays[i[0]],"to",weekdays[i[1]],"]")


#Finding which all variables appear in the tree ...

feature = regressor.tree_.feature
new_nums = list(filter(lambda x: x >=0, feature))
T = [features[i] for i in list(set(new_nums))]
print("Variable(s) that appear in the tree is/are:", T)
print("Variable(s) that didn't appear in the tree is/are:", list(set(features)-set(T)))



#Which variables are important? ...

important = regressor.feature_importances_
#print(important)
i = list(np.nonzero(important > 0.099))
T = [features[j] for j in i[0]]
print("Most important variables are:", T)



#Computing the MSE ...

print("MSE of regression tree with maximum depth possible: %.3f" %mean_squared_error(y_test, regressor.predict(X_test)))


#Estimating best depth for the regression tree ...

mse = []
for depth in range(1,depth_max+1):
    regressor_ = DecisionTreeRegressor(max_depth = depth)
    regressor_.fit(X_train, y_train)
    mse.append(mean_squared_error(y_test, regressor_.predict(X_test)))

depth_best = mse.index(min(mse))+1
print("Best depth of the regression tree is", depth_best)

#Learn the final regression tree with restriction on depth ...

regressor_f = DecisionTreeRegressor(max_depth = depth_best)
regressor_f.fit(X_train, y_train) 
print("\n")

print("\nNOTE: For plotting the learnt decision tree classifier uncomment the\
 lines in the script!\n")
#..........................................................#
#..........................................................#
#.............Uncomment MULTILINE COMMENT here.............#
#..........................................................#
#..........................................................#
#Plotting the regression tree ...
'''
dot_data = export_graphviz(regressor_f, out_file=None, feature_names=features,                            filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data) 
conda_fix(graph)
Image(graph.create_png())
'''

#Estimation of total no. of nodes and leaves in the decision tree ...

estimator = regressor_f
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
leaves = 0
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True
        leaves = leaves + 1
print('Total No. of nodes = ', n_nodes)
print('Total No. of leaves = ', leaves)
depth_max = max(node_depth)
print("Input depth of tree is", depth_best,"and calculated depth of tree is", depth_max)

#Finding into how many different groups of days does the tree divide the data ...
weekdays = ["Monday", "Tuesday", "Wednesday", "Thrusday", "Friday", "Saturday", "Sunday"]
wd_index = 4
thresh = []
for i in range(n_nodes):
    if feature[i] == wd_index:
        thresh.append(threshold[i])

thresh = list(set(thresh))
thresh.sort()
print("Threshold values for weekdays:", thresh)

groups = []
for i in thresh:
    a = int(i)
    groups.append([0,a])
    groups.append([a+1,6])

b_set = set(tuple(x) for x in groups)
group = [ list(x) for x in b_set ]
group.sort(key = lambda x: groups.index(x) )
num = len(group)
print("Hence, clearly the tree divides the data in totally", num,"groups of weekdays namely:")
for i in group:
    if i[0] == i[1]:
        print("[",weekdays[i[0]],"]")
    else:
        print("[",weekdays[i[0]],"to",weekdays[i[1]],"]")


#Finding which all variables appear in the tree ...

feature = regressor_f.tree_.feature
new_nums = list(filter(lambda x: x >=0, feature))
T = [features[i] for i in list(set(new_nums))]
print("Variable(s) that appear in the tree is/are:", T)
print("Variable(s) that didn't appear in the tree is/are:", list(set(features)-set(T)))


#Which variables are important? ...

important = regressor_f.feature_importances_
#print(important)
i = list(np.nonzero(important > 0.099))
T = [features[j] for j in i[0]]
print("Most important variables are:", T)



#Computing the MSE ...

print("MSE of regression tree with best depth: %.3f" %mean_squared_error(y_test, regressor_f.predict(X_test)))
print("MSE of regression tree with maximum depth possible: %.3f" %mean_squared_error(y_test, regressor.predict(X_test)))



print("........................PART 2..........................")

#Recoding the "month" variable ...

bikes['month'].replace([1,2],1,inplace=True)
bikes['month'].replace([5,6,7,8,9,10],2,inplace=True)
bikes['month'].replace([3,4,11,12],3,inplace=True)

train_data = bikes.loc[:]
X = train_data.loc[:, "season":"windspeed"]
y = train_data.loc[:, "count"]

#Train and test data split ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=None)

#Normalizing regression target ...

y_train = (y_train-min(y_train))/(max(y_train)-min(y_train))
y_test = (y_test-min(y_test))/(max(y_test)-min(y_test))

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#Learn regression tree without and restriction on max depth ...

regressor2 = DecisionTreeRegressor()
regressor2.fit(X_train, y_train)

#Extracting features list from the dataset ...

b = bikes.columns.get_values()
features = np.delete(b, 0)
features = np.delete(features, 10)

print("Features are:",features)


#Estimation of total no. of nodes and leaves in the decision tree without restriction...

estimator = regressor2
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
leaves = 0
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True
        leaves = leaves + 1
print('Total No. of nodes = ', n_nodes)
print('Total No. of leaves = ', leaves)
depth_max = max(node_depth)
print("Depth of tree without any restriction on the maximum depth is", depth_max,".")



#Finding into how many different groups of days does the tree divide the data ...
weekdays = ["Monday", "Tuesday", "Wednesday", "Thrusday", "Friday", "Saturday", "Sunday"]
wd_index = 4
thresh = []
for i in range(n_nodes):
    if feature[i] == wd_index:
        thresh.append(threshold[i])

thresh = list(set(thresh))
thresh.sort()
print("Threshold values for weekdays:", thresh)


groups = []
for i in thresh:
    a = int(i)
    groups.append([0,a])
    groups.append([a+1,6])

b_set = set(tuple(x) for x in groups)
group = [ list(x) for x in b_set ]
group.sort(key = lambda x: groups.index(x) )
num = len(group)
print("Hence, clearly the tree divides the data in totally", num,"groups of weekdays namely:")
for i in group:
    if i[0] == i[1]:
        print("[",weekdays[i[0]],"]")
    else:
        print("[",weekdays[i[0]],"to",weekdays[i[1]],"]")



#Finding which all variables appear in the tree ...

feature = regressor2.tree_.feature
new_nums = list(filter(lambda x: x >=0, feature))
T = [features[i] for i in list(set(new_nums))]
print("Variable(s) that appear in the tree is/are:", T)
print("Variable(s) that didn't appear in the tree is/are:", list(set(features)-set(T)))



#Which variables are important? ...

important = regressor2.feature_importances_
#print(important)
i = list(np.nonzero(important > 0.099))
T = [features[j] for j in i[0]]
print("Most important variables are:", T)



#Computing the MSE ...

print("MSE of regression tree with maximum depth possible: %.3f" %mean_squared_error(y_test, regressor2.predict(X_test)))



#Estimating best depth for the regression tree ...

mse = []
for depth in range(1,depth_max+1):
    regressor2_ = DecisionTreeRegressor(max_depth = depth)
    regressor2_.fit(X_train, y_train)
    mse.append(mean_squared_error(y_test, regressor2_.predict(X_test)))

depth_best = mse.index(min(mse))+1
print("Best depth of the regression tree is", depth_best)

#Learn the final regression tree with restriction on depth ...

regressor2_f = DecisionTreeRegressor(max_depth = depth_best)
regressor2_f.fit(X_train, y_train) 
print("\n")

print("\nNOTE: For plotting the learnt decision tree classifier uncomment the\
 lines in the script!\n")
#..........................................................#
#..........................................................#
#.............Uncomment MULTILINE COMMENT here.............#
#..........................................................#
#..........................................................#
#Plotting the regression tree ...
'''
dot_data = export_graphviz(regressor2_f, out_file=None, feature_names=features,                            filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data) 
conda_fix(graph)
Image(graph.create_png())
'''
#Estimation of total no. of nodes and leaves in the decision tree ...

estimator = regressor2_f
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
leaves = 0
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True
        leaves = leaves + 1
print('Total No. of nodes = ', n_nodes)
print('Total No. of leaves = ', leaves)
depth_max = max(node_depth)
print("Input depth of tree is", depth_best,"and calculated depth of tree is", depth_max)



#Finding into how many different groups of days does the tree divide the data ...
weekdays = ["Monday", "Tuesday", "Wednesday", "Thrusday", "Friday", "Saturday", "Sunday"]
wd_index = 4
thresh = []
for i in range(n_nodes):
    if feature[i] == wd_index:
        thresh.append(threshold[i])

thresh = list(set(thresh))
thresh.sort()
print("Threshold values for weekdays:", thresh)


groups = []
for i in thresh:
    a = int(i)
    groups.append([0,a])
    groups.append([a+1,6])

b_set = set(tuple(x) for x in groups)
group = [ list(x) for x in b_set ]
group.sort(key = lambda x: groups.index(x) )
num = len(group)
print("Hence, clearly the tree divides the data in totally", num,"groups of weekdays namely:")
for i in group:
    if i[0] == i[1]:
        print("[",weekdays[i[0]],"]")
    else:
        print("[",weekdays[i[0]],"to",weekdays[i[1]],"]")



#Finding which all variables appear in the tree ...

feature = regressor2_f.tree_.feature
new_nums = list(filter(lambda x: x >=0, feature))
T = [features[i] for i in list(set(new_nums))]
print("Variable(s) that appear in the tree is/are:", T)
print("Variable(s) that didn't appear in the tree is/are:", list(set(features)-set(T)))



#Which variables are important? ...

important = regressor2_f.feature_importances_
#print(important)
i = list(np.nonzero(important > 0.099))
T = [features[j] for j in i[0]]
print("Most important variables are:", T)



#Computing the MSE ...

print("MSE of regression tree with best depth: %.3f" %mean_squared_error(y_test, regressor2_f.predict(X_test)))
print("MSE of regression tree with maximum depth possible: %.3f" %mean_squared_error(y_test, regressor2.predict(X_test)))

