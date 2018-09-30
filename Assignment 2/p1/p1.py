#Importing required libraries ...

import os
import sys
import numpy as np
import collections
from sklearn import metrics
from sklearn.externals.six import StringIO 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus
from matplotlib import pyplot as plt

#Loading train and test data with labels ...

train_data = np.genfromtxt('trainX.csv',delimiter=',')
test_data = np.genfromtxt('testX.csv',delimiter=',')

train_label = np.genfromtxt('trainY.csv',delimiter=',')
test_label = np.genfromtxt('testY.csv',delimiter=',')

#print(train_data.shape)
#print(test_data.shape)

#Building decision tree classifier on given training dataset ... 

dtree = DecisionTreeClassifier(criterion = "entropy")
dtree.fit(train_data, train_label)

#Reporting test accuracy and populating confusion matrix ... 

expected = test_label
predicted = dtree.predict(test_data)
print("Total Test Accuracy = %.2f" %metrics.accuracy_score(expected, predicted))
print("Missclassification rate of Malignant Class= %.2f" %(1-metrics.precision_score(expected, predicted)))
print("Missclassification rate of Benign Class= %.2f\n" %(1-((57*metrics.accuracy_score(expected, predicted)-\
                                                             32*metrics.precision_score(expected, predicted))/25)))
print('Confusion Matrix:')
print('[True Malignant  False Malignant]')
print('[False Benign    True Benign    ]\n')
print(metrics.confusion_matrix(expected, predicted))

#List of features ... 

features = ['Avg Radius', 'Avg Texture', 'Avg Perimeter', 'Avg Area', 'Avg Smoothness', 'Avg Compactness', 'Avg Concavity', \
            'Avg Number of concave portions of contour', 'Avg Symmetry', 'Avg Fractal dimension', \
            'StD Radius', 'StD Texture', 'StD Perimeter', 'StD Area', 'StD Smoothness', 'StD Compactness', 'StD Concavity', \
            'StD Number of concave portions of contour', 'StD Symmetry', 'StD Fractal dimension', \
            'Max Radius', 'Max Texture', 'Max Perimeter', 'Max Area', 'Max Smoothness', 'Max Compactness', 'Max Concavity', \
            'Max Number of concave portions of contour', 'Max Symmetry', 'Max Fractal dimension']

#Function to fix the problem with compatibility of graphviz with conda ...

def conda_fix(graph):
        path = os.path.join(sys.base_exec_prefix, "Library", "bin", "graphviz")
        paths = ("dot", "twopi", "neato", "circo", "fdp")
        paths = {p: os.path.join(path, "{}.exe".format(p)) for p in paths}
        graph.set_graphviz_executables(paths)

#Plotting the learnt decision tree classifier ...
print("\nNOTE: For plotting the learnt decision tree classifier uncomment the\
 lines in the script!\n")
#..........................................................#
#..........................................................#
#.............Uncomment MULTILINE COMMENT here.............#
#..........................................................#
#..........................................................#
'''
dot_data = export_graphviz(dtree, out_file=None, feature_names=features, class_names=['malignant', 'benign'], \
                           filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data) 
colors = ('lightblue', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
conda_fix(graph)
Image(graph.create_png())
'''
#..........................................................#
#..........................................................#
#.............Uncomment MULTILINE COMMENT here.............#
#..........................................................#
#..........................................................#

#Calculating no. of nodes and no. of leaves in the learnt decision tree ...

n_nodes = dtree.tree_.node_count
children_left = dtree.tree_.children_left
children_right = dtree.tree_.children_right
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
        leaves = leaves + 1
print('Total No. of nodes = ', n_nodes)

print('Total No. of leaves = ', leaves)

#Randomly sampling 10%, 20%, ..., 100% of the training data and builing corresponding decision tree classifier ... 
#Recording for each of them train and test accuracies ... 

train_accuracy = np.zeros(10)
test_accuracy = np.zeros(10)
for i in range(1,11):
    print("Randomly sampling %d0" %i, "% of training data..." )
    t = np.random.choice(455, (455*i)//10)
    #print(t)
    x = train_data[t]
    y = train_label[t]
    #Fit Tree
    dtree = DecisionTreeClassifier(criterion = "entropy")
    dtree.fit(x, y)
    #Train Accuracy
    expected = y
    predicted = dtree.predict(x)
    train_accuracy[i-1] = metrics.accuracy_score(expected, predicted)
    print("Train Accuracy = %.2f" %train_accuracy[i-1])
    #Test Accuracy
    expected = test_label
    predicted = dtree.predict(test_data)
    test_accuracy[i-1] = metrics.accuracy_score(expected, predicted)
    print("Test Accuracy = %.2f\n" %test_accuracy[i-1])
    
    print("Average Test Accuracy = %.2f" %np.average(test_accuracy))

#Plot to show variation of training and test accuracies with number of training samples ...

percentage = ["10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"]
x = np.array([1,2,3,4,5,6,7,8,9,10])
plt.xticks(x, percentage)
plt.plot(x, train_accuracy,'m', label='train_accuracy')
plt.scatter(x, train_accuracy, cmap=plt.cm.get_cmap('cubehelix', 1))
plt.plot(x, test_accuracy,'c', label='test_accuracy')
plt.scatter(x, test_accuracy, cmap=plt.cm.get_cmap('cubehelix', 1))
plt.ylim((0,1.05))

legend = plt.legend(loc='lower left', shadow=True)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title("Train and Test Accuracy v/s Number of training samples")

plt.show()
