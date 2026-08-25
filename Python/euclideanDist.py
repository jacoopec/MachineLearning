
def compute_eu_dist(a: np.ndarray,b: np.ndarray):
    N,M = A.shape
    assert  b.shape[0] == M
    distances = np.zeros((N,))
    
    for i in range(N):
        for j in range(M):
            distances[i] = (A[i,j] - b[j])**2
        distances[i] = distances[i]**0.5
    return distances

def compute_eu_dist_numpy(a: np.ndarray,b: np.ndarray):
    N,M = A.shape
    assert  b.shape[0] == M
    return np.sqrt(np.power((A-B[np.newaxis, :]),2).sum(1)) 


A = np.array([[5,3,43,2,4],[3,6,3,4,3],[7,4,3,2,8],[9,2,1,3,5]])

B = np.array([4,2,1,4,1])

rst =  compute_eu_dist(A,B)

print(A)

