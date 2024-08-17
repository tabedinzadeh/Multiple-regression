
import matplotlib.pyplot as plt 
import pandas as pd 
import pylab as pl 
import numpy as np 
import csv

df = pd.read_csv('automobil.csv')
print(df.describe())

cdf=df[['length', 'height', 'width', 'weight']]
print(cdf)

plt.scatter(df.width ,df.weight , color='blue')
plt.xlabel('width')
plt.ylabel('weight')
plt.show()

plt.scatter(df.height ,df.weight , color='blue')
plt.xlabel('height')
plt.ylabel('weight')
plt.show()

plt.scatter(df.length ,df.weight , color='blue')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

msk = np.random.rand(len(df)) <0.8
train = cdf[msk]
test = cdf[~msk]

from sklearn import linear_model
reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['length', 'width', 'height']])
train_y = np.asanyarray(train[['weight']])
reg.fit ( train_x , train_y)

print('coefs:', reg.coef_)
print('inter:' , reg.intercept_)

from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['length', 'width', 'height']])
test_y = np.asanyarray(test[['weight']])
test_y_= reg.predict(test_x)


print(" mean ab error: %.2f" % np.mean( np.absolute(test_y_ - test_y))) 
print(" residule sum of sq: %.2f(MSE)" % np.mean((test_y_ - test_y)**2)) 
print(" R2_score: %.2f" % r2_score(test_y , test_y_))
print(' variance score: %.2f' % reg.score(test_x, test_y))