#Loading Libraries
import numpy as np
import matplotlib.pyplot as plt

#Loading Dataset
print('\nLoading Dataset...')
train_data = np.genfromtxt('Wage_dataset.csv',delimiter=',')
train_data.shape

wage = train_data[0:3000,10]
age = train_data[0:3000,1]
year = train_data[0:3000,0]
education = train_data[0:3000,4]
education.shape

r = np.matrix(wage)
r = r.T


#################
#Education vs wage using mean squared error for each data point
print('\nEducation vs Wage...')
print('Predicting best polynomial order(k)...')
print('Using MSE between each wage[i] and estimate of f(education[i]) as metric')
def education_mse(o):
    N = wage.size
    k=o #polynomial order
    D = np.zeros((N,k+1))
    for i in range(N):
        for j in range(k+1):
            D[i,j] = education[i]**j

    D = np.matrix(D)

    w = np.linalg.inv(D.T*D)*D.T*r

    x = education
    y = wage
    x = np.arange(0.0, 6.0, 1.0)
    z = np.zeros(x.size)
    MSE = 0
    for i in range(x.size):
        X = np.zeros(k+1)
        for j in range(k+1):
            X[j] = x[i]**(j)
        X = np.matrix(X)
        z[i] = X*w
        MSE += (y[i]-z[i])**2
    MSE = np.sqrt(MSE)
    return MSE

n = 10 #values to test till
mse = np.zeros(n)
print('Mean Square Error vs Polynomial Order')
for o in range(1,n):
    mse[o] = education_mse(o)
    print('MSE = ',mse[o],' for polynomial order = ',o)
plt.title('Mean Square Error vs Polynomial Order')
plt.scatter(np.arange(1,n,1),mse[1:])
plt.show()

#Best k value
m = list(mse[1:])
k = m.index(min(m))+1
print('Best polynomial order is ',k,' with MSE equal to ',m[k-1])
N = wage.size
D = np.zeros((N,k+1))
for i in range(N):
    for j in range(k+1):
        D[i,j] = education[i]**j

D = np.matrix(D)

w = np.linalg.inv(D.T*D)*D.T*r
print('Learned w is: ')
print(w)
x = education
y = wage

plt.scatter(x, y)

x = np.arange(0.0, 6.0, 1.0)
y = np.zeros(x.size)
for i in range(x.size):
    X = np.zeros(k+1)
    for j in range(k+1):
        X[j] = x[i]**(j)
    X = np.matrix(X)
    y[i] = X*w
plt.plot(x,y,'k')

##Average Values
n = np.zeros(6)
for i in range(N):
    temp = int(education[i])
    n[temp]+=1

sum = np.zeros(6)
for i in range(N):
    temp = int(education[i])
    sum[temp]+=wage[i]

avg = np.zeros(6)
for i in range(6):
    if n[i]!=0: avg[i] = sum[i]/n[i]

x = np.arange(1,6,1)
y = avg[1:]
plt.scatter(x, y, marker='*', s=150)
plt.title('Estimate of Deterministic Function f(x) = (w.T)*x')
plt.show()

#####################

#Education vs wage using mean squared error for average(wage[education[j]])
print('\nEducation vs Wage...')
print('Predicting best polynomial order(k)...')
print('Using MSE between each wage[i] and average(estimate of f(wage[education[j]]) as metric')
##Average Values
n = np.zeros(6)
for i in range(N):
    temp = int(education[i])
    n[temp]+=1

sum = np.zeros(6)
for i in range(N):
    temp = int(education[i])
    sum[temp]+=wage[i]

avg = np.zeros(6)
for i in range(6):
    if n[i]!=0: avg[i] = sum[i]/n[i]

def EDU(o):  
    #Education
    #Assuming Linear Model i.e. k = 1
    N = wage.size
    k=o
    D = np.zeros((N,k+1))
    for i in range(N):
        for j in range(k+1):
            D[i,j] = education[i]**j

    D = np.matrix(D)

    w = np.linalg.inv(D.T*D)*D.T*r

    x = education
    y = wage

    x = np.arange(0.0, 6.0, 1.0)
    y = np.zeros(x.size)
    for i in range(x.size):
        X = np.zeros(k+1)
        for j in range(k+1):
            X[j] = x[i]**(j)
        X = np.matrix(X)
        y[i] = X*w

    x = np.arange(1,6,1)
    y = avg[1:]
    
    MSE = 0
    for i in x:
        X = np.zeros(k+1)
        for j in range(k+1):
            X[j] = i**(j)
        X = np.matrix(X)
        y = X*w
        MSE += (y-avg[i])**2
    MSE = np.sqrt(MSE)
    return MSE

n = 10 #values to test till
print('Mean Square Error vs Polynomial Order')
mse = np.zeros(n)
for o in range(1,n):
    mse[o] = EDU(o)
    print('MSE = ',mse[o],' for polynomial order = ',o)
plt.title('Mean Square Error vs Polynomial Order')
plt.scatter(np.arange(1,n,1),mse[1:])
plt.show()

#Best k value
m = list(mse[1:])
k = m.index(min(m))+1
print('Best polynomial order is ',k,' with MSE equal to ',m[k-1])
N = wage.size
D = np.zeros((N,k+1))
for i in range(N):
    for j in range(k+1):
        D[i,j] = education[i]**j

D = np.matrix(D)

w = np.linalg.inv(D.T*D)*D.T*r
print('Learned w is: ')
print(w)
x = education
y = wage

plt.scatter(x, y)

x = np.arange(0.0, 6.0, 1.0)
y = np.zeros(x.size)
for i in range(x.size):
    X = np.zeros(k+1)
    for j in range(k+1):
        X[j] = x[i]**(j)
    X = np.matrix(X)
    y[i] = X*w
plt.plot(x,y,'k')
x = np.arange(1,6,1)
y = avg[1:]
plt.scatter(x, y, marker='*', s=150)
plt.title('Estimate of Deterministic Function f(x) = (w.T)*x')
plt.show()

########################

#Age vs wage using mean squared error for each data point
print('\nAge vs Wage...')
print('Predicting best polynomial order(k)...')
print('Using MSE between each wage[i] and estimate of f(education[i]) as metric')
def age_mse(o):
    N = wage.size
    k=o #polynomial order
    D = np.zeros((N,k+1))
    for i in range(N):
        for j in range(k+1):
            D[i,j] = age[i]**j

    D = np.matrix(D)

    w = np.linalg.inv(D.T*D)*D.T*r

    x = age
    y = wage
    x = np.arange(0.0, 100.0, 1.0)
    z = np.zeros(x.size)
    MSE = 0
    for i in range(x.size):
        X = np.zeros(k+1)
        for j in range(k+1):
            X[j] = x[i]**(j)
        X = np.matrix(X)
        z[i] = X*w
        MSE += (y[i]-z[i])**2
    MSE = np.sqrt(MSE)
    return MSE

n = 10 #values to test till
print('Mean Square Error vs Polynomial Order')
mse = np.zeros(n)
for o in range(1,n):
    mse[o] = age_mse(o)
    print('MSE = ',mse[o],' for polynomial order = ',o)
plt.title('Mean Square Error vs Polynomial Order')
plt.scatter(np.arange(1,n,1),mse[1:])
plt.show()

#Best k value
m = list(mse[1:])
k = m.index(min(m))+1
print('Best polynomial order is ',k,' with MSE equal to ',m[k-1])
N = wage.size
D = np.zeros((N,k+1))
for i in range(N):
    for j in range(k+1):
        D[i,j] = age[i]**j

D = np.matrix(D)

w = np.linalg.inv(D.T*D)*D.T*r
print('Learned w is:')
print(w)
x = age
y = wage

plt.scatter(x, y)

x = np.arange(0.0, 100.0, 1.0)
y = np.zeros(x.size)
for i in range(x.size):
    X = np.zeros(k+1)
    for j in range(k+1):
        X[j] = x[i]**(j)
    X = np.matrix(X)
    y[i] = X*w
plt.plot(x,y,'k')

##Average Values
n = np.zeros(81)
for i in range(N):
    temp = int(age[i])
    n[temp]+=1

sum = np.zeros(81)
for i in range(N):
    temp = int(age[i])
    sum[temp]+=wage[i]
  
avg = np.zeros(81)
for i in range(81):
    if n[i]!=0: avg[i] = sum[i]/n[i]

x = np.arange(18,81,1)
y = avg[18:]
plt.scatter(x, y, marker='*', s=150)
plt.title('Estimate of Deterministic Function f(x) = (w.T)*x')
plt.show()

####################

#Age vs wage using mean squared error for average(wage[education[j]])
print('\nAge vs Wage...')
print('Predicting best polynomial order(k)...')
print('Using MSE between each wage[i] and average(estimate of f(wage[education[j]]) as metric')
##Average Values
n = np.zeros(81)
for i in range(N):
    temp = int(age[i])
    n[temp]+=1

sum = np.zeros(81)
for i in range(N):
    temp = int(age[i])
    sum[temp]+=wage[i]
  
avg = np.zeros(81)
for i in range(81):
    if n[i]!=0: avg[i] = sum[i]/n[i]
        
def AGE(o):
    #Age
    N = wage.size
    k=o
    D = np.zeros((N,k+1))
    for i in range(N):
        for j in range(k+1):
            D[i,j] = age[i]**j

    D = np.matrix(D)
    w = np.linalg.inv(D.T*D)*D.T*r

    x = age
    y = wage

    x = np.arange(0.0, 100.0, 1.0)
    y = np.zeros(x.size)
    for i in range(x.size):
        X = np.zeros(k+1)
        for j in range(k+1):
            X[j] = x[i]**(j)
        X = np.matrix(X)
        y[i] = X*w

    x = np.arange(18,81,1)
    y = avg[18:]

    #mean squared error
    MSE = 0
    for i in x:
        X = np.zeros(k+1)
        for j in range(k+1):
            X[j] = i**(j)
        X = np.matrix(X)
        y = X*w
        MSE += (y-avg[i])**2
    MSE = np.sqrt(MSE)
    return MSE

n = 10 #values to test till
mse = np.zeros(n)
print('Mean Square Error vs Polynomial Order')
for o in range(1,n):
    mse[o] = AGE(o)
    print('MSE = ',mse[o],' for polynomial order = ',o)
plt.title('Mean Square Error vs Polynomial Order')
plt.scatter(np.arange(1,n,1),mse[1:])
plt.show()

#Best k value
m = list(mse[1:])
k = m.index(min(m))+1
print('Best polynomial order is ',k,' with MSE equal to ',m[k-1])
N = wage.size
D = np.zeros((N,k+1))
for i in range(N):
    for j in range(k+1):
        D[i,j] = age[i]**j

D = np.matrix(D)

w = np.linalg.inv(D.T*D)*D.T*r
print('Learned w is:')
print(w)
x = age
y = wage

plt.scatter(x, y)

x = np.arange(0.0, 100.0, 1.0)
y = np.zeros(x.size)
for i in range(x.size):
    X = np.zeros(k+1)
    for j in range(k+1):
        X[j] = x[i]**(j)
    X = np.matrix(X)
    y[i] = X*w
plt.plot(x,y,'k')

x = np.arange(18,81,1)
y = avg[18:]
plt.scatter(x, y, marker='*', s=150)
plt.title('Estimate of Deterministic Function f(x) = (w.T)*x')
plt.show()

##################

#Year vs wage using mean squared error for each data point
print('\nYear vs Wage...')
print('Predicting best polynomial order(k)...')
print('Using MSE between each wage[i] and estimate of f(education[i]) as metric')
#Year
def year_mse(o):
    N = wage.size
    k=o #polynomial order
    D = np.zeros((N,k+1))
    for i in range(N):
        for j in range(k+1):
            D[i,j] = year[i]**j

    D = np.matrix(D)

    w = np.linalg.inv(D.T*D)*D.T*r

    x = year
    y = wage
    x = np.arange(2001, 2011, 1.0)
    z = np.zeros(x.size)
    MSE = 0
    for i in range(x.size):
        X = np.zeros(k+1)
        for j in range(k+1):
            X[j] = x[i]**(j)
        X = np.matrix(X)
        z[i] = X*w
        MSE += (y[i]-z[i])**2
    MSE = np.sqrt(MSE)
    return MSE

n = 10 #values to test till
mse = np.zeros(n)
print('Mean Square Error vs Polynomial Order')
for o in range(1,n):
    mse[o] = year_mse(o)
    print('MSE = ',mse[o],' for polynomial order = ',o)
plt.title('Mean Square Error vs Polynomial Order')
plt.scatter(np.arange(1,n,1),mse[1:])
plt.show()

#Best k value
m = list(mse[1:])
k = m.index(min(m))+1
print('Best polynomial order is ',k,' with MSE equal to ',m[k-1])
N = wage.size
D = np.zeros((N,k+1))
for i in range(N):
    for j in range(k+1):
        D[i,j] = year[i]**j

D = np.matrix(D)

w = np.linalg.inv(D.T*D)*D.T*r
print('Learned w is:')
print(w)
x = year
y = wage

plt.scatter(x, y)

x = np.arange(2001, 2011, 1.0)
y = np.zeros(x.size)
for i in range(x.size):
    X = np.zeros(k+1)
    for j in range(k+1):
        X[j] = x[i]**(j)
    X = np.matrix(X)
    y[i] = X*w
plt.plot(x,y,'k')

##Average Values
n = np.zeros(2010)
for i in range(N):
    temp = int(year[i])
    n[temp]+=1

sum = np.zeros(2010)
for i in range(N):
    temp = int(year[i])
    sum[temp]+=wage[i]
  
avg = np.zeros(2010)
for i in range(2010):
    if n[i]!=0: avg[i] = sum[i]/n[i]

x = np.arange(2003,2010,1)
y = avg[2003:]
plt.scatter(x, y, marker='*', s=150)
plt.title('Estimate of Deterministic Function f(x) = (w.T)*x')
plt.show()
#####################

#Year vs wage using mean squared error for average(wage[education[j]])
print('\nYear vs Wage...')
print('Predicting best polynomial order(k)...')
print('Using MSE between each wage[i] and average(estimate of f(wage[education[j]]) as metric')
##Average Values
n = np.zeros(2010)
for i in range(N):
    temp = int(year[i])
    n[temp]+=1

sum = np.zeros(2010)
for i in range(N):
    temp = int(year[i])
    sum[temp]+=wage[i]
  
avg = np.zeros(2010)
for i in range(2010):
    if n[i]!=0: avg[i] = sum[i]/n[i]
        
def YEAR(o):
    #Year
    N = wage.size
    k=o
    D = np.zeros((N,k+1))
    for i in range(N):
        for j in range(k+1):
            D[i,j] = year[i]**j

    D = np.matrix(D)
    w = np.linalg.inv(D.T*D)*D.T*r

    x = year
    y = wage

    x = np.arange(2001, 2011, 1.0)
    y = np.zeros(x.size)
    for i in range(x.size):
        X = np.zeros(k+1)
        for j in range(k+1):
            X[j] = x[i]**(j)
        X = np.matrix(X)
        y[i] = X*w

    x = np.arange(2003,2010,1)
    y = avg[2003:]

    #mean squared error
    MSE = 0
    for i in x:
        X = np.zeros(k+1)
        for j in range(k+1):
            X[j] = i**(j)
        X = np.matrix(X)
        y = X*w
        MSE += (y-avg[i])**2
    MSE = np.sqrt(MSE)    
    return MSE

n = 10 #values to test till
mse = np.zeros(n)
print('Mean Square Error vs Polynomial Order')
for o in range(1,n):
    mse[o] = YEAR(o)
    print('MSE = ',mse[o],' for polynomial order = ',o)
plt.title('Mean Square Error vs Polynomial Order')
plt.scatter(np.arange(1,n,1),mse[1:])
plt.show()

#Best k value
m = list(mse[1:])
k = m.index(min(m))+1
print('Best polynomial order is ',k,' with MSE equal to ',m[k-1])
N = wage.size
D = np.zeros((N,k+1))
for i in range(N):
    for j in range(k+1):
        D[i,j] = year[i]**j

D = np.matrix(D)

w = np.linalg.inv(D.T*D)*D.T*r
print('Learned w is:')
print(w)
x = year
y = wage

plt.scatter(x, y)

x = np.arange(2001, 2011, 1.0)
y = np.zeros(x.size)
for i in range(x.size):
    X = np.zeros(k+1)
    for j in range(k+1):
        X[j] = x[i]**(j)
    X = np.matrix(X)
    y[i] = X*w
plt.plot(x,y,'k')

x = np.arange(2003,2010,1)
y = avg[2003:]
plt.scatter(x, y, marker='*', s=150)
plt.title('Estimate of Deterministic Function f(x) = (w.T)*x')
plt.show()
################
