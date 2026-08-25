import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":
    a = np.array([[1,2,1,0,0.2],
                  [1,2,1,0,0.2],
                  [1,2,1,0,0.2],
                  [1,2,1,0,0.2],
                  [1,2,1,0,0.2]])


    plt.scatter(points[:,0], points[:,1], c=labels, cmap='viridis', marker='o')
    plt.axvline(x=best_split_info[0]["threshold"], linestyle="--")
    plt.axhline(y=best_split_info[1]["threshold"], linestyle="--")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Points")
    plt.grid(True)
    plt.show()