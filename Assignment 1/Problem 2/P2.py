#Note: Keep closing current generated plot for further execution of the script!
#Loading Libraries
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#Loading Dataset
print('\nLoading Dataset...')
train_data = np.genfromtxt('P2_train.csv',delimiter=',')
test_data = np.genfromtxt('P2_test.csv',delimiter=',')
x = train_data[0:310,0:2]
y = train_data[0:310,2]
train_label = y
train_data = x
x = test_data[0:310,0:2]
y = test_data[0:310,2]
test_label = y
test_data = x
print('Number of training samples: ',train_label.shape)
print('Number of testing samples: ',test_label.shape)

#Estimating p1 and p2
print('\nEstimating p0 and p1...')
num_train = train_label.size
p1 = 0
p2 = 0
for i in range(num_train):
    if train_label[i] == 0.0:
        p1+=1
    elif train_label[i] == 1.0:
        p2+=1

print("p0 = p =",p1/num_train)
print("p1 = 1-p =",p2/num_train)

#Testing data class 1 & 2 size
num_test = test_label.size
t1 = 0
t2 = 0
for i in range(num_test):
    if test_label[i] == 0.0:
        t1+=1
    elif test_label[i] == 1.0:
        t2+=1

#Mean estimation
print('\nEstimating Mean_0 and Mean_1...')
sum1 = 0
sum2 = 0
for i in range(num_train):
    if train_label[i] == 0.0:
        sum1+=train_data[i,:]
    elif train_label[i] == 1.0:
        sum2+=train_data[i,:]

mean1 = sum1/p1
mean2 = sum2/p2
print('Estimate of Mean_0 =')
print(mean1)
print('Estimate of Mean_1 =')
print(mean2)

#Plotting Dataset with Estimated Means
x = train_data[0:310,0]
y = train_data[0:310,1]

plt.scatter(x, y,c=train_label, cmap=plt.cm.get_cmap('flag', 2))
plt.colorbar(ticks=range(2))

x = [mean1[0],mean2[0]]
y = [mean1[1],mean2[1]]
plt.scatter(x, y, cmap=plt.cm.get_cmap('brg', 2))
plt.axis('equal')
plt.title('Training Datapoints with Estimated Means', loc='left')
plt.show()

#Covariance Matrices Estimation
print('\nEstimating Σ0 and Σ1...')
sigma1 = np.zeros((2,2))
sigma2 = np.zeros((2,2))


for i in range(2):
    for j in range(2):
        for t in range(310):
            if train_label[t] == 0.0:
                sigma1[i,j] += (train_data[t,i]-mean1[i])*(train_data[t,j]-mean1[j])
sigma1 /= p1

for i in range(2):
    for j in range(2):
        for t in range(310):
            if train_label[t] == 1.0:
                sigma2[i,j] += (train_data[t,i]-mean2[i])*(train_data[t,j]-mean2[j])
sigma2 /= p2
print('Estimate of Σ0 =')
print(sigma1)
print('Estimate of Σ1 =')
print(sigma2)

#Plotting Dataset with Estimated Means & Covariance Matrices
x = train_data[0:310,0]
y = train_data[0:310,1]

plt.scatter(x, y,c=train_label, cmap=plt.cm.get_cmap('flag', 2))
plt.colorbar(ticks=range(2))

x = [mean1[0],mean2[0]]
y = [mean1[1],mean2[1]]
plt.scatter(x, y, cmap=plt.cm.get_cmap('brg', 2))

x, y = np.mgrid[-4:6:.1, -8:8:.1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal(mean1, sigma1)
plt.contour(x, y, rv1.pdf(pos))

rv2 = multivariate_normal(mean2, sigma2)
plt.contour(x, y, rv2.pdf(pos))

plt.contour(x, y, np.sign(rv1.pdf(pos)-rv2.pdf(pos)))
plt.axis('equal')
plt.title('Learned Iso-Probability Contours with Estimated Parameters', loc='left')


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

S1 = np.linalg.det(C1)
S2 = np.linalg.det(C2)

C1inv = np.linalg.inv(C1)
C2inv = np.linalg.inv(C2)
F1 = 0
F2 = 0

for i in range(N):
    x = train_data[i,:]
    X = np.matrix(x)
    X = np.matrix.transpose(X)
    y1 = ((X-M1).T)*(C1inv)*(X-M1)
    y2 = ((X-M2).T)*(C2inv)*(X-M2) 
    G1 = np.log(P1)+(-0.5*(y1+np.log(S1)))
    G2 = np.log(P2)+(-0.5*(y2+np.log(S2))) 
    
    if G1>G2:
        label = 0.0
    elif G1<G2:
        label = 1.0
            
    if train_label[i] != label:
        if train_label[i]==0.0:
            F1+=1
        elif train_label[i]==1.0:
            F2+=1
F = F1+F2
accuracy = (N-F)/N
print("\nTraining Accuracy: ",accuracy)
print("Class 0 Accuracy: ", (N1-F1)/N1)
print("Class 1 Accuracy: ", (N2-F2)/N2)
Confusion = np.matrix([[N1-F1,F1],[F2,N2-F2]])
print("\nConfusion Matrix: \n")
CM = [['True Class 0', 'False Class 1'],['False Class 0', 'True Class 1']]
print(CM)
print(Confusion)
plt.show()

#Classification using Σ0 and Σ1
print('\nClassification using estimated Σ0 and Σ1...')
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
    
    if G1>G2:
        label = 0.0
    elif G1<G2:
        label = 1.0
            
    if test_label[i] != label:
        if test_label[i]==0.0:
            false1+=1
        elif test_label[i]==1.0:
            false2+=1
false = false1+false2
accuracy = (num_test-false)/num_test
print("Test Accuracy: ",accuracy)
print("Class 0 Accuracy: ", (t1-false1)/t1)
print("Class 1 Accuracy: ", (t2-false2)/t2)

x = test_data[0:90,0]
y = test_data[0:90,1]

plt.scatter(x, y,c=test_label, cmap=plt.cm.get_cmap('flag', 2))
plt.colorbar(ticks=range(2))

x, y = np.mgrid[-4:6:.1, -12:8:.1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal(mean1, sigma1)
plt.contour(x, y, rv1.pdf(pos))

rv2 = multivariate_normal(mean2, sigma2)
plt.contour(x, y, rv2.pdf(pos))

plt.contour(x, y, np.sign(rv1.pdf(pos)-rv2.pdf(pos)))
plt.axis('equal')
plt.title('Plot of discriminant function and isoprobability contours using estimated covariance matrices', loc='left')

Confusion = np.matrix([[t1-false1,false1],[false2,t2-false2]])
print("\nConfusion Matrix: \n")
CM = [['True Class 0', 'False Class 1'],['False Class 0', 'True Class 1']]
print(CM)
print(Confusion)
plt.show()

print('\nDifferent variations of covariance matrices:')
#Case A.
print('\nA. Classification using Σ0 = Σ1 = Σ of the form')
print('a 0')
print('0 a')
a = (np.random.randint(9))+1
D1 = D2 = np.matrix([[a,0],[0,a]])
print('Σ = ')
print(D1)

R1 = np.linalg.det(D1)
R2 = np.linalg.det(D2)

D1inv = np.linalg.inv(D1)
D2inv = np.linalg.inv(D2)
false1 = 0
false2 = 0
for i in range(num_test):
    x = test_data[i,:]
    X = np.matrix(x)
    X = np.matrix.transpose(X)
    y1 = ((X-M1).T)*(D1inv)*(X-M1)
    y2 = ((X-M2).T)*(D2inv)*(X-M2)
    G1 = np.log(P1)-(0.5*y1)
    G2 = np.log(P2)-(0.5*y2) 
    if G1>=G2:
        label = 0.0
    else:
        label = 1.0
    if label != test_label[i]:
        if test_label[i]==0.0:
            false1+=1
        elif test_label[i]==1.0:
            false2+=1
false = false1+false2
accuracy = (num_test-false)/num_test
print("\nTest Accuracy: ",accuracy)
print("Class 0 Accuracy: ", (t1-false1)/t1)
print("Class 1 Accuracy: ", (t2-false2)/t2)


#Plotting
x = test_data[0:90,0]
y = test_data[0:90,1]

plt.scatter(x, y,c=test_label, cmap=plt.cm.get_cmap('flag', 2))
plt.colorbar(ticks=range(2))

x, y = np.mgrid[-10:10:.1, -12:8:.1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal(mean1, D1)
plt.contour(x, y, rv1.pdf(pos))

rv2 = multivariate_normal(mean2, D2)
plt.contour(x, y, rv2.pdf(pos))

plt.contour(x, y, np.sign(rv1.pdf(pos)-rv2.pdf(pos)))

plt.axis('equal')
plt.title('Plot of discriminant function and isoprobability contours for Case A', loc='left')

Confusion = np.matrix([[t1-false1,false1],[false2,t2-false2]])
print("\nConfusion Matrix: \n")
CM = [['True Class 0', 'False Class 1'],['False Class 0', 'True Class 1']]
print(CM)
print(Confusion)
plt.show()

#Case B.
print('\nB. Classification using Σ0 = Σ1 = Σ of the form')
print('a 0')
print('0 b')
a = (np.random.randint(9))+1
b = (np.random.randint(9))+1
D1 = D2 = np.matrix([[a,0],[0,b]])
print('Σ = ')
print(D1)

R1 = np.linalg.det(D1)
R2 = np.linalg.det(D2)

D1inv = np.linalg.inv(D1)
D2inv = np.linalg.inv(D2)
false1 = 0
false2 = 0
for i in range(num_test):
    x = test_data[i,:]
    X = np.matrix(x)
    X = np.matrix.transpose(X)
    y1 = ((X-M1).T)*(D1inv)*(X-M1)
    y2 = ((X-M2).T)*(D2inv)*(X-M2) 
    G1 = np.log(P1)-(0.5*y1)
    G2 = np.log(P2)-(0.5*y2) 
    if G1>=G2:
        label = 0.0
    else:
        label = 1.0
    if label != test_label[i]:
        if test_label[i]==0.0:
            false1+=1
        elif test_label[i]==1.0:
            false2+=1
false = false1+false2
accuracy = (num_test-false)/num_test
print("\nTest Accuracy: ",accuracy)
print("Class 0 Accuracy: ", (t1-false1)/t1)
print("Class 1 Accuracy: ", (t2-false2)/t2)


#Plotting
x = test_data[0:90,0]
y = test_data[0:90,1]

plt.scatter(x, y,c=test_label, cmap=plt.cm.get_cmap('flag', 2))
plt.colorbar(ticks=range(2))

x, y = np.mgrid[-10:10:.1, -12:8:.1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal(mean1, D1)
plt.contour(x, y, rv1.pdf(pos))

rv2 = multivariate_normal(mean2, D2)
plt.contour(x, y, rv2.pdf(pos))

plt.contour(x, y, np.sign(rv1.pdf(pos)-rv2.pdf(pos)))

plt.axis('equal')
plt.title('Plot of discriminant function and isoprobability contours for Case B', loc='left')


Confusion = np.matrix([[t1-false1,false1],[false2,t2-false2]])
print("Confusion Matrix: \n")
CM = [['True Class 0', 'False Class 1'],['False Class 0', 'True Class 1']]
print(CM)
print(Confusion)
plt.show()

#Case C.
print('\nC. Classification using Σ0 = Σ1 = Σ of the form')
print('a b')
print('c d')
a = (np.random.randint(9))+1
b = (np.random.randint(9))+1
rho = ((np.random.randint(99))+1)/100
D1 = D2 = np.matrix([[a,rho*((a*b)**(0.5))],[rho*((a*b)**(0.5)),b]])
print('Σ = ')
print(D1)

R1 = np.linalg.det(D1)
R2 = np.linalg.det(D2)

D1inv = np.linalg.inv(D1)
D2inv = np.linalg.inv(D2)
false1 = 0
false2 = 0
for i in range(num_test):
    x = test_data[i,:]
    X = np.matrix(x)
    X = np.matrix.transpose(X)
    y1 = ((X-M1).T)*(D1inv)*(X-M1)
    y2 = ((X-M2).T)*(D2inv)*(X-M2) 
    G1 = np.log(P1)-(0.5*y1)
    G2 = np.log(P2)-(0.5*y2)
    if G1>=G2:
        label = 0.0
    else:
        label = 1.0
    if label != test_label[i]:
        if test_label[i]==0.0:
            false1+=1
        elif test_label[i]==1.0:
            false2+=1
false = false1+false2
accuracy = (num_test-false)/num_test
print("Test Accuracy: ",accuracy)
print("Class 0 Accuracy: ", (t1-false1)/t1)
print("Class 1 Accuracy: ", (t2-false2)/t2)


#Plotting
x = test_data[0:90,0]
y = test_data[0:90,1]

plt.scatter(x, y,c=test_label, cmap=plt.cm.get_cmap('flag', 2))
plt.colorbar(ticks=range(2))

x, y = np.mgrid[-10:10:.1, -12:8:.1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal(mean1, D1)
plt.contour(x, y, rv1.pdf(pos))

rv2 = multivariate_normal(mean2, D2)
plt.contour(x, y, rv2.pdf(pos))

plt.contour(x, y, np.sign(rv1.pdf(pos)-rv2.pdf(pos)))

plt.axis('equal')
plt.title('Plot of discriminant function and isoprobability contours for Case C', loc='left')


Confusion = np.matrix([[t1-false1,false1],[false2,t2-false2]])
print("Confusion Matrix: \n")
CM = [['True Class 0', 'False Class 1'],['False Class 0', 'True Class 1']]
print(CM)
print(Confusion)
plt.show()

#Case D.
print('\nD. Classification using different Σ0 and Σ1 of the form')
print('a b')
print('c d')
a1 = (np.random.randint(9))+1
b1 = (np.random.randint(9))+1
a2 = (np.random.randint(9))+1
b2 = (np.random.randint(9))+1
rho = ((np.random.randint(99))+1)/100
D1 = np.matrix([[a1,rho*((a1*b1)**(0.5))],[rho*((a1*b1)**(0.5)),b1]])
D2 = np.matrix([[a2,rho*((a2*b2)**(0.5))],[rho*((a2*b2)**(0.5)),b2]])
print('Σ1 = ')
print(D1)
print('Σ2 = ')
print(D2)

R1 = np.linalg.det(D1)
R2 = np.linalg.det(D2)

D1inv = np.linalg.inv(D1)
D2inv = np.linalg.inv(D2)
false1 = 0
false2 = 0
for i in range(num_test):
    x = test_data[i,:]
    X = np.matrix(x)
    X = np.matrix.transpose(X)
    y1 = ((X-M1).T)*(D1inv)*(X-M1)
    y2 = ((X-M2).T)*(D2inv)*(X-M2) 
    G1 = np.log(P1)-(0.5*y1)-(0.5*np.log(R1))
    G2 = np.log(P2)-(0.5*y2)-(0.5*np.log(R2)) 
    if G1>=G2:
        label = 0.0
    else:
        label = 1.0
    if label != test_label[i]:
        if test_label[i]==0.0:
            false1+=1
        elif test_label[i]==1.0:
            false2+=1
false = false1+false2
accuracy = (num_test-false)/num_test
print("Test Accuracy: ",accuracy)
print("Class 0 Accuracy: ", (t1-false1)/t1)
print("Class 1 Accuracy: ", (t2-false2)/t2)


#Plotting
x = test_data[0:90,0]
y = test_data[0:90,1]

plt.scatter(x, y,c=test_label, cmap=plt.cm.get_cmap('flag', 2))
plt.colorbar(ticks=range(2))

x, y = np.mgrid[-10:10:.1, -12:8:.1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal(mean1, D1)
plt.contour(x, y, rv1.pdf(pos))

rv2 = multivariate_normal(mean2, D2)
plt.contour(x, y, rv2.pdf(pos))

plt.contour(x, y, np.sign(rv1.pdf(pos)-rv2.pdf(pos)))

plt.axis('equal')
plt.title('Plot of discriminant function and isoprobability contours for Case D', loc='left')


Confusion = np.matrix([[t1-false1,false1],[false2,t2-false2]])
print("Confusion Matrix: \n")
CM = [['True Class 0', 'False Class 1'],['False Class 0', 'True Class 1']]
print(CM)
print(Confusion)
plt.show()
