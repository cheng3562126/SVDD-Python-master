# -*- coding: utf-8 -*-
"""

An example for parameter optimization using PSO.

"scikit-opt" is required in this examples.

https://github.com/guofei9987/scikit-opt

"""

import sys

import numpy as np

sys.path.append("..")
from src.BaseSVDD import BaseSVDD, BananaDataset
from sko.PSO import PSO
import matplotlib.pyplot as plt


# Banana-shaped dataset generation and partitioning
X, y = BananaDataset.generate(number=200, display='on')
X_train, X_test, y_train, y_test = BananaDataset.split(X, y, ratio=0.3)


# objective function

aa = np.empty([0, 2])

def objective_func(x):
    x1, x2 = x
    svdd = BaseSVDD(C=x1, gamma=x2, kernel='rbf', display='off')
    y = 1-svdd.fit(X_train, y_train).accuracy
    global aa

    aa = np.append(aa, [[x1, x2]], axis=0)
    return y


# Do PSO
pso = PSO(func=objective_func, n_dim=2, pop=10, max_iter=20,
          lb=[0.01, 0.01], ub=[1, 3], w=0.8, c1=0.5, c2=0.5)
pso.run()
# print("aa  =  ", aa)

svdd = BaseSVDD(C=aa[0][0],gamma=aa[0][1], kernel='rbf', display='off')
svdd.fit(X_train,  y_train)
svdd.plot_boundary(X_train,  y_train)

svdd = BaseSVDD(C=aa[1][0],gamma=aa[1][1], kernel='rbf', display='off')
svdd.fit(X_train,  y_train)
svdd.plot_boundary(X_train,  y_train)

svdd = BaseSVDD(C=aa[2][0],gamma=aa[2][1], kernel='rbf', display='off')
svdd.fit(X_train,  y_train)
svdd.plot_boundary(X_train,  y_train)

print('best_x is', pso.gbest_x)
print('best_y is', pso.gbest_y)
#svdd.fit(X_train,  y_train)

svdd = BaseSVDD(C=pso.gbest_x[0],gamma=pso.gbest_x[1], kernel='rbf', display='off')
svdd.fit(X_train,  y_train)
svdd.plot_boundary(X_train,  y_train)

#
#svdd.plot_boundary(X_train,  y_train)
# plot the result
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
ax.plot(pso.gbest_y_hist)
ax.yaxis.grid()
plt.show()
