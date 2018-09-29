#Loading Libraries
import numpy as np
from matplotlib import pyplot as plt

#Loading Dataset
print('\nLoading Dataset...')
train_data = np.genfromtxt('P1_data_train.csv',delimiter=',')
test_data = np.genfromtxt('P1_data_test.csv',delimiter=',')
train_label = np.genfromtxt('P1_labels_train.csv',delimiter=',')
test_label = np.genfromtxt('P1_labels_test.csv',delimiter=',')
print('Number of training samples: ',train_label.shape)
print('Number of testing samples: ',test_label.shape)

#Estimating π5 and π6
print('\nEstimating π5 and π6...')
num_train = train_label.size
p1 = 0
p2 = 0
for i in range(num_train):
    if train_label[i] == 5.0:
        p1+=1
    elif train_label[i] == 6.0:
        p2+=1
print("π5 = ",p1/num_train)
print("π6 = ",p2/num_train)

#Testing data class 5 & 6 size
num_test = test_label.size
t1 = 0
t2 = 0
for i in range(num_test):
    if test_label[i] == 5.0:
        t1+=1
    elif test_label[i] == 6.0:
        t2+=1

#Evaluating μ5 and μ6
print('\nEstimating Mean_5 and Mean_6...')
sum1 = 0
sum2 = 0
for i in range(num_train):
    if train_label[i] == 5.0:
        sum1+=train_data[i,:]
    elif train_label[i] == 6.0:
        sum2+=train_data[i,:]
mean1 = sum1/p1
mean2 = sum2/p2

#Evaluating Σ5 and Σ6
print('\nEstimating Σ5 and Σ6...')
sigma1 = np.zeros((64,64))
sigma2 = np.zeros((64,64))
for i in range(64):
    for j in range(64):
        for t in range(777):
            if train_label[t] == 5.0:
                sigma1[i,j] += (train_data[t,i]-mean1[i])*(train_data[t,j]-mean1[j])
            elif train_label[t] == 6.0:
                sigma2[i,j] += (train_data[t,i]-mean2[i])*(train_data[t,j]-mean2[j])
                
sigma1 /= (p1)            
sigma2 /= (p2)

#Summarizing
C1 = np.matrix(sigma1)
C2 = np.matrix(sigma2)
M1 = np.matrix(mean1)
M1 = np.matrix.transpose(M1)
M2 = np.matrix(mean2)
M2 = np.matrix.transpose(M2)
N1 = p1
N2 = p2
N = N1+N2
P1 = N1/N
P2 = N2/N
#print(M1,M2)
#print(C1,C2)
print('\nNumber of Class 5 datapoints in training set = ', N1)
print('Number of Class 6 datapoints in training set = ', N2)
print('Number of Class 5 datapoints in testing set = ', t1)
print('Number of Class 6 datapoints in testing set = ', t2)

#Classification using Bayes Decison Criteria
#For estimated Σ5 and Σ6
print('\nClassification of test data using Bayes decision criterion...')
print('\nA.')
print('For estimated Σ5 and Σ6')
S1 = np.linalg.det(C1)
S2 = np.linalg.det(C2)

C1inv = np.linalg.inv(C1)
C2inv = np.linalg.inv(C2)
false1 = 0
false2 = 0
for i in range(num_test):
    x = test_data[i,:]
    X = np.matrix(x)
    X = np.matrix.transpose(X)
    y1 = ((X-M1).T)*(C1inv)*(X-M1)
    y2 = ((X-M2).T)*(C2inv)*(X-M2)
    
    G1 = np.log(P1)+(-0.5*(y1+np.log(S1)))
    G2 = np.log(P2)+(-0.5*(y2+np.log(S2)))
    
    if G1 > G2:
        label = 5.0
    else:
        label = 6.0
    
    if label != test_label[i]:
        if test_label[i]==5.0:
            false1+=1
        elif test_label[i]==6.0:
            false2+=1
    
false = false1+false2
accuracy = (num_test-false)/num_test
print("Accuracy = ",accuracy)
print("Class 5 Accuracy = ", (t1-false1)/t1)
print("Class 6 Accuracy = ", (t2-false2)/t2)
#Confusion Matrix
print('\nConfusion Matrix:')
CM = [['True Class 5', 'False Class 6'],['False Class 5', 'True Class 6']]
print(CM)
Confusion = np.matrix([[t1-false1,false1],[false2,t2-false2]])
print(Confusion)

#For same Σ
print('\nB.')
print('For Σ = (π5*Σ5)+(π6*Σ6)')
sigma = (P1*sigma1)+(P2*sigma2)
C = np.matrix(sigma)
Cinv = np.linalg.inv(C)
false1 = 0
false2 = 0
for i in range(num_test):
    x = test_data[i,:]
    X = np.matrix(x)
    X = np.matrix.transpose(X)
    y1 = ((X-M1).T)*(Cinv)*(X-M1)
    y2 = ((X-M2).T)*(Cinv)*(X-M2) 
    G1 = np.log(P1)-(0.5*y1)
    G2 = np.log(P2)-(0.5*y2)
    if G1 > G2:
        label = 5.0
    else:
        label = 6.0
    if label != test_label[i]:
        if test_label[i]==5.0:
            false1+=1
        elif test_label[i]==6.0:
            false2+=1
false = false1+false2
accuracy = (num_test-false)/num_test
print("Accuracy = ",accuracy)
print("Class 1 Accuracy = ", (t1-false1)/t1)
print("Class 2 Accuracy = ", (t2-false2)/t2)
#Confusion Matrix
print('\nConfusion Matrix:')
CM = [['True Class 5', 'False Class 6'],['False Class 5', 'True Class 6']]
print(CM)
Confusion = np.matrix([[t1-false1,false1],[false2,t2-false2]])
print(Confusion)

#For same Σ
print('\nC.')
print('For Σ = a*I (I = Identity Matrix)')
a = (np.random.randint(9))+1
#print(a)
sigma = a*np.eye(64)
C = np.matrix(sigma)
Cinv = np.linalg.inv(C)
false1 = 0
false2 = 0
for i in range(num_test):
    x = test_data[i,:]
    X = np.matrix(x)
    X = np.matrix.transpose(X)
    y1 = ((X-M1).T)*(Cinv)*(X-M1)
    y2 = ((X-M2).T)*(Cinv)*(X-M2) 
    G1 = np.log(P1)-(0.5*y1)
    G2 = np.log(P2)-(0.5*y2)
    if G1 > G2:
        label = 5.0
    else:
        label = 6.0
    if label != test_label[i]:
        if test_label[i]==5.0:
            false1+=1
        elif test_label[i]==6.0:
            false2+=1
false = false1+false2
accuracy = (num_test-false)/num_test
print("Accuracy = ",accuracy)
print("Class 1 Accuracy = ", (t1-false1)/t1)
print("Class 2 Accuracy = ", (t2-false2)/t2)
#Confusion Matrix
print('\nConfusion Matrix:')
CM = [['True Class 5', 'False Class 6'],['False Class 5', 'True Class 6']]
print(CM)
Confusion = np.matrix([[t1-false1,false1],[false2,t2-false2]])
print(Confusion)
