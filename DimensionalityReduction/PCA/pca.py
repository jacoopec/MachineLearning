import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, functions needs samples as columns
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # eigenvectors v = [:, i] column vector, transpose this for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors

        idxs = np.argsort(eigenvalues)[::-1]
        print("idxs")
        print(idxs)
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        print("eigenvectors")
        print(eigenvectors)
        print("eigenvalues")
        print(eigenvalues)

        self.components = eigenvectors[:self.n_components]
        print(self.components.shape)
        print(self.components)


    def transform(self, X):
        # projects data
        X = X - self.mean
        return np.dot(X, self.components.T)


# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn import datasets

    # data = datasets.load_digits()
    # data = datasets.load_iris()
    
    # data = np.array([[0,1,2],
    #     [2,4,-2],
    #     [0,1,2],
    #     [8,6,4],
    #     [8,0,5]])
    data = np.array([[0,1],
        [2,4],
        [10,-1],
        [8,-3],
        [-12,3],
        [8,6],
        [8,0]]
    )
    
    y = [0,1,1,0,1,0,1]

    # X = data.data
    X = data
    
    # y = data.target

    pca = PCA(2)
    main_eigen_vctrs = pca.fit(X)
    print("Vectors: ", pca.components)
    
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)
    
    print("X_projected ",X_projected)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    
    print("x1 ", x1)
    print("x2 ", x2)

    plt.scatter(x1, x2, c = y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
    
    origin = np.array([[0, 0]])  # PCA-transformed origin
    components_2D = pca.components[:2]  # already 2D because n_components=2

    for vec in components_2D:
        plt.arrow(
            0, 0,               # start
            vec[0], vec[1],     # direction
            color="red",
            width=0.01,
            head_width=0.05
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()