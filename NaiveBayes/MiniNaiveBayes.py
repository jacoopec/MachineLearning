import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def plotDistributions(means, vars):
    print(means)
    print(vars)
    means = means.flatten()
    variances = vars.flatten()
    print("print")
    print(means)
    print(variances)
    x = np.linspace(-100, 100, 100)
    stds = np.sqrt(variances)
    print(stds)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()

    
    for i, (m, s) in enumerate(zip(means, stds)):
        y = 1/(s * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - m) / s)**2)

        ax = axes[i]
        ax.plot(x, y)
        ax.set_title(f"Gaussian {i+1}\nMean={m:.2f}, Var={variances[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.show()


X_train = np.array([[1.2,2],[10.2,20],[43.2,40],[66.3,70]])
y_train = np.array([0,0,1,1])
X_test  = np.array([[53.4,58],[10.2,12],[40.7,50],[6.3,6]])
y_test  = np.array([0,0,1,1])

classes = np.unique(y_train)
n_classes = 2
n_samples, n_features = X_train.shape

means  = np.zeros((n_classes, n_features), dtype = np.float64)
vars   = np.zeros((n_classes, n_features), dtype = np.float64)
priors = np.zeros(n_classes,               dtype = np.float64)

print(means)
print(vars)

# train
for idx, c in enumerate(classes):
    X_c = X_train[y_train == c]
    print(X_c.mean(axis = 0))
    means[idx, :] = X_c.mean(axis = 0)
    vars[idx, :]  = X_c.var(axis = 0)
    priors[idx]  = X_c.shape[0] / float(n_samples)
        
        
def pdf(idx, x):    
    _mean = means[idx]
    _var = vars[idx]
    numerator = np.exp(-((x - _mean) ** 2) / (2 * _var))
    denominator = np.sqrt(2 * np.pi * _var)
    return numerator / denominator

# predict
for x in X_test:
    posteriors = []
    for idx, c in enumerate(classes):
        prior     = np.log(priors[idx])
        posterior = np.sum(np.log(pdf(idx, x)))
        posterior = posterior + prior
        posteriors.append(posterior)
    
print("predictions",np.argmax(posteriors))
print(means)
print(vars)
plotDistributions(means, vars)
        
        
