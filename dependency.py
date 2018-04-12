import pandas as pd
import numpy as np
import re
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
define the functions may be used
"""
# Define local weighted linear regression function
# lower-case x is the point where the weight is the biggest
# upper-case X is the training data Y is the label of the training data
# if LS is TRUE we calculate the least squares regression (without wheughts)
def lwl_regression(x, X, Y, LS=False, k=1.0):
    ones = np.ones(len(X))
    X = np.array([X.T[0],X.T[1],ones]).T
    m = len(X)
    W = np.matrix(np.zeros((m,m)))
    x = np.array([x[0][0],x[0][1],1])
    for i in range(m):
        xi = np.array(X[i])
        if not LS:
            W[i, i] = exp(np.linalg.norm(x - xi) / (-2 * k ** 2))
        else:
            W[i, i] = 1
    xwx = X.T * W * X
    if np.linalg.det(xwx) == 0:
        print('xWx is a singular matrix')
        return
    w = xwx.I * X.T * W * Y
    return w

# how to use the trained model
def h(X,w):
    ones = np.ones(len(X))
    X = np.array([X.T[0], X.T[1], ones]).T
    return X * w

# build training set and testig set
def dataset(X):
    train = X.sample(int(len(X) / 1.5))
    test = X.drop(train.index.values)
    return train, test

# find the monthly data of China treasury bond index
bond = pd.read_csv("./data/bond.csv")
bond = bond.values
final_data = []
for j in range(len(bond) - 2):
    i = j + 1
    if re.match("(..)/*", bond[i][0]).group() == re.match("(..)/*", bond[i+1][0]).group() and re.match("(..)/*", bond[i][0]).group() != re.match("(..)/*", bond[i-1][0]).group():
        data = bond[i].tolist()
        final_data.append(data)
bond = pd.DataFrame(final_data, columns=["date", "code", "name", "close", "high",
                                         "low", "opem", "pre-close", "variation", "variation-rate", "volumn", "amount"])


"""
dataset:
independent variables <-- change rate of monthly CPI, and change rate of M2 money supply (monthly) 
dependent variables <-- change rate of the index of China treasury bond (monthly)
"""

# build the dataset
cpi = pd.read_csv("./data/cpi.csv")
cpi_data = cpi["cpi"][:len(bond)-1]
cpi_data = ((cpi_data[:-1].values - cpi_data[1:].values) / cpi_data[1:].values) 
m2 = pd.read_csv("./data/money_supply.csv")
m2 = m2["m2"][:len(bond)-1].astype(float) 
m2 = ((m2[:-1].values - m2[1:].values) / m2[1:].values) 
bond_data = bond["close"][1:]
bond_data = ((bond_data[:-1].values - bond_data[1:].values) / bond_data[1:].values) 
data = np.array([m2, cpi_data, bond_data]).astype(float)

# correlation analysis
cov = np.cov(data, bias=1)  # covariance matrix
corr = np.corrcoef(data)  # correlation coefficient matrix
print("covariance matrix of M2, CPI and the index of China treasury bond：")
print(cov)
print("correlation coefficient matrix of M2, CPI and the index of China treasury bond：")
print(corr)

# separate the dataset into training set and testing set 
X = pd.DataFrame(data.T.tolist(), columns=["m2", "cpi", "bond"])
training, testing = dataset(X)

# reshape the testing set
training_X = np.array(training[["m2", "cpi"]].values)
training_Y = np.array(training[["bond"]].values)
testing_X = np.array(testing[["m2", "cpi"]].values)
testing_Y = np.array(testing[["bond"]].values)
X1 = X[["m2", "cpi"]]
x = np.array(X1.loc[[8]].values)
theta = lwl_regression(x, training_X, training_Y, LS=True)

hx = h(testing_X,theta)

# draw the data of the testing set
xxx = np.array(testing[["m2"]].values)
yyy = np.array(testing[["cpi"]].values)
zzz = testing_Y
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(xxx, yyy, zzz, marker = '.')

# draw the surface of the theta(parameters of the linear regression)
xx = np.array([-0.05, -0.05, 0.05, 0.05])
yy = np.array([-0.05, 0.05, -0.05, 0.05])
x1 = np.array([[-0.05, -0.05, 0.05, 0.05], [-0.05, 0.05, -0.05, 0.05]]).T
xx, yy = np.meshgrid(xx, yy)

zz = np.array(xx * np.float(theta[0][0]) + yy * np.float(theta[1][0]) + np.float(theta[2][0]))
ax.plot_surface(xx, yy,
                zz.reshape(xx.shape), color="grey")
print(theta)
ax.set_xlabel('M2')
ax.set_ylabel('CPI')
ax.set_zlabel('Treasury')
plt.show()
