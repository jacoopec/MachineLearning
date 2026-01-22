import numpy as np


a = np.array([[-0.67477416, -1.36821115],
                [ 2.72135134, -3.06043585],
                [-2.27026861, -0.19651188],
                [-0.98948172, -0.55386429],
                [ 1.04324278,  0.69021444],
                [-1.15879713, -0.69561762],
                [-1.78519736,  0.13748657],
                [ 1.12894049,  0.58323204],
                [-1.04738099, -0.51331923],
                [-0.87130666, -1.58158263]])
b = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])

_mean   = np.zeros((2, 2), dtype = np.float64)
_var    = np.zeros((2, 2), dtype = np.float64)
_priors = np.zeros(2,      dtype = np.float64)



firstCol = a[:, 0]
secCol = a[:, 1]

_mean[0, :] = firstCol.mean(axis = 0)
_var[0, :] = firstCol.var(axis = 0)
_priors[0] = firstCol.shape[0] / float(10)

_mean[1, :] = secCol.mean(axis = 0)
_var[1, :] = secCol.var(axis = 0)
_priors[1] = secCol.shape[0] / float(10)


mean1 = self._mean[class_idx]
var1 = self._var[class_idx]
numerator1 = np.exp(-((x - mean) ** 2) / (2 * var))
denominator1 = np.sqrt(2 * np.pi * var)

mean2 = _mean[1, :]
var3 = _var[1, :]
numerator2 = np.exp(-((x - mean) ** 2) / (2 * var))
denominator2 = np.sqrt(2 * np.pi * var)

