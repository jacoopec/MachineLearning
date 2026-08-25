import matplotlib.pyplot as plt
import numpy as np

def best_split(points, labels):
    best_split = None
    best_threshold = [None] * data.shape[1]
    best_error = float("inf") * data.shape[1]
    best_split_info = {
            "threshold": None,
            "left_class": None,
            "right_class": None,
            "error": None,
            "left_indices": None,
            "right_indices": None,
        }  * data.shape[1]
    
    for i in range(points.shape[1]):

        thresholds = (points[:,i][:-1] + points[:,i][1:]) / 2
        

        # print(points)
        
        
        for threshold in thresholds:
            left = labels[points[:,i] <= threshold]
            right = labels[points[:,i] > threshold]
            print(f"Feature {i}, Threshold {threshold}:")
            # print(left)
            # print(right)

            if len(left) == 0 or len(right) == 0:
                    continue
                
            left_class = majority_class(left)
            right_class = majority_class(right)
        
            # print(f"Left class: {left_class}, Right class: {right_class}")
        

            predictions = np.where(points[:,i] <= threshold, left_class, right_class)
            error = np.mean(predictions != labels)

            if error < best_error[i]:
                best_error[i] = error
                best_threshold[i] = threshold
                best_split_info[i]={
                    "threshold": threshold,
                    "left_class": left_class,
                    "right_class": right_class,
                    "error": error,
                    "left_indices": np.where(points[:,i] <= threshold)[0],
                    "right_indices": np.where(points[:,i] > threshold)[0],
                }
                
                print(best_split_info)

        
    return best_split_info[np.argmin(best_error)]

def majority_class(values):
    classes, counts = np.unique(values, return_counts=True)
    return classes[np.argmax(counts)]

if __name__ == "__main__":
    
    # Example 2D points: (x, y)
    # data = np.array([[1, 2], [2.5,3], [3,4], [2, 3], [3, 1], [8, 5],[6,6], [9, 6],[7,8]])
    # labels = np.array([0, 0, 1, 0, 0, 1,0, 1, 1]) 
    
    data = np.array([[1.     ,    2.        ],
        [2.5        ,3.        ],
        [3.         ,4.        ],
        [2.         ,3.        ],
        [3.         ,1.        ],
        [8.         ,5.        ],
        [6.         ,6.        ],
        [9.         ,6.        ],
        [7.         ,8.        ],
        [1.06094342 ,1.79200318],
        [2.65009024 ,3.18811294],
        [2.60979296 ,3.7395641 ],
        [2.02556808 ,2.93675148],
        [2.99663977 ,0.82939121],
        [8.17587959 ,5.15555839],
        [6.01320614 ,6.22544824],
        [9.09350187 ,5.82814151],
        [7.07375016 ,7.80822348],
        [1.17569006 ,1.99001482],
        [2.46302753 ,2.86381409],
        [3.24450827 ,3.9690941 ],
        [1.91433444 ,2.92957329],
        [3.10646184 ,1.07308881],
        [8.08254652 ,5.0861642 ],
        [6.42832952 ,5.918717  ],
        [8.89755145 ,5.83724545],
        [7.12319588 ,8.22579446],
        [0.97721051 ,1.8319687 ],
        [2.33510376 ,3.13011856],
        [3.14865083 ,4.10863085],
        [1.86689806 ,3.04643226],
        [3.02333716 ,1.04373772],
        [8.17428576 ,5.04471911],
        [6.13578271 ,6.01351581],
        [9.05782388 ,6.12625765],
        [6.70856884 ,7.93606576]])
    labels = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0,0 ])
    n_of_features = data.shape[1]
    points = data[:, :n_of_features] 

    # print(points[:,1])
    # thresholds = (points[:,0][:-1] + points[:,0][1:]) / 2
    # print(thresholds)
    
    # print(data.shape[0])
    # print(points)
    # print(points.shape)  
    # print(points[:,1])

    # print(points[1])
    


    best_split_ = best_split(points, labels)
    # print(best_split_info[1])

    plt.scatter(points[:,0], points[:,1], c=labels, cmap='viridis', marker='o')
    plt.axvline(x=best_split_info[0]["threshold"], linestyle="--")
    plt.axhline(y=best_split_info[1]["threshold"], linestyle="--")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Points")
    plt.grid(True)
    plt.show()