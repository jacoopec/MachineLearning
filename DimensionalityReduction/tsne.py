import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------
# 1. Compute pairwise squared distances
# -----------------------------------------
def pairwise_distances(X):
    sum_X = np.sum(np.square(X), axis=1)
    return -2 * np.dot(X, X.T) + sum_X[:, None] + sum_X[None, :]


# -------------------------------------------------------------
# 2. Compute conditional probabilities p(j|i) using a Gaussian
# -------------------------------------------------------------
def compute_p_conditional(D, sigma):
    P = np.exp(-D / (2 * sigma ** 2))
    np.fill_diagonal(P, 0)   # remove self-similarity
    P /= np.sum(P, axis=1, keepdims=True)
    return P


# -------------------------------------------------------------
# 3. Compute symmetric joint probabilities P_ij
# -------------------------------------------------------------
def compute_p_matrix(X, sigma=1.0):
    D = pairwise_distances(X)
    P_cond = compute_p_conditional(D, sigma)
    P = (P_cond + P_cond.T) / (2 * X.shape[0])
    return P


# -------------------------------------------------------------
# 4. Compute q_ij in low-dimensional space (Student t-distribution)
# -------------------------------------------------------------
def compute_q_matrix(Y):
    D = pairwise_distances(Y)
    Q = 1 / (1 + D)
    np.fill_diagonal(Q, 0)
    Q /= np.sum(Q)
    return Q, D


# -------------------------------------------------------------
# 5. Gradient of KL divergence between P and Q
# -------------------------------------------------------------
def tsne_gradient(P, Q, Y, D):
    n, dim = Y.shape
    grad = np.zeros((n, dim))
    
    PQ = P - Q
    
    for i in range(n):
        for j in range(n):
            grad[i] += 4 * PQ[i, j] * (Y[i] - Y[j]) / (1 + D[i, j])
    
    return grad


# -------------------------------------------------------------
# 6. t-SNE main routine
# -------------------------------------------------------------
def tsne(X, dim=2, lr=200, iterations=300, sigma=1.0):
    n = X.shape[0]
    
    # Compute high-dimensional similarities
    P = compute_p_matrix(X, sigma)
    
    # Initialize low-dimensional map randomly
    Y = np.random.randn(n, dim) * 1e-4
    
    for it in range(iterations):
        Q, D_y = compute_q_matrix(Y)
        grad = tsne_gradient(P, Q, Y, D_y)
        
        Y -= lr * grad
        
        if it % 50 == 0:
            cost = np.sum(P * np.log((P + 1e-8) / (Q + 1e-8)))
            print(f"Iteration {it}, KL divergence = {cost:.4f}")
    
    return Y


# -------------------------------------------------------------
# 7. Example usage with your simple dataset
# -------------------------------------------------------------
data = {
    "feature1": [1, 2, 2, 8, 9, 9, 1, 2, 3],
    "feature2": [1, 1, 2, 8, 9, 8, 3, 2, 3],
    "class":    [0, 0, 0, 1, 1, 1, 2, 2, 2]
}

X = np.column_stack((data["feature1"], data["feature2"]))
y = np.array(data["class"])

Y = tsne(X, dim=2, iterations=300)

print("Final low-dimensional coordinates:\n", Y)

plt.scatter(Y[:,0], Y[:,1], c=y, cmap="Set1")
for i in range(len(Y)):
    plt.text(Y[i,0], Y[i,1], str(y[i]))
plt.title("t-SNE (from scratch)")
plt.show()