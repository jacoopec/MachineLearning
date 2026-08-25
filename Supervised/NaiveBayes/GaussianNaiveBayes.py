import numpy as np

# Since the features were binary:  free ∈ {0,1}, meeting ∈ {0,1}
# I use Bernoulli Naive Bayes

# important | beautiful | done | missing
a = np.array([[9.3,0.5],
              [8.2,-0.9],
              [13.1,3.2],
              [10.0,0.5],
              [11.3,1.5],
              [9.3,2.5],
              [12.3,-3.2],
              [10.1,-1.5],
              [10.9,-2.9],
              [10.6,-1.5]])
b = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])

clss  = [np.count_nonzero(b == 0)/b.shape[0], np.count_nonzero(b == 1)/b.shape[0]]

# feats = np.array([[np.count_nonzero(a[:,0]==1)/clss[0],np.count_nonzero(a[:,1]==1)/clss[0],np.count_nonzero(a[:,2]==1)/clss[0],np.count_nonzero(a[:,3]==1)/clss[0]],
#                   [np.count_nonzero(a[:,0]==0)/clss[1],np.count_nonzero(a[:,1]==0)/clss[1],np.count_nonzero(a[:,2]==0)/clss[1],np.count_nonzero(a[:,3]==0)/clss[1]]
#                   ])

_mean   = np.zeros((2, 2), dtype = np.float64)
_var    = np.zeros((2, 2), dtype = np.float64)
_priors = clss



firstCol = a[:, 0]
secCol   = a[:, 1]

_mean[0, :] = firstCol.mean(axis = 0)
_var[0, :]  = firstCol.var(axis = 0)

_mean[1, :] = secCol.mean(axis = 0)
_var[1, :]  = secCol.var(axis = 0)


mean = _mean[1, :]
var = _var[1, :]
x =  np.array([10.5,-1])
res = np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)

print(res)

