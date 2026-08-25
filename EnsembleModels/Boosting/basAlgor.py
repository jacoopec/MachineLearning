import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier

dataset = np.array([[0.5,1,10],
                    [1.5,0,20],
                    [0.2,1,15],
                    [-0.3,1,30],
                    [0.4,0,2],
                    [0.5,1,30],
                    [-0.6,0,2]])

labels = np.array([1,1,1,0,0,0,0])

weights =  np.full_like(labels, fill_value=1/labels.shape[0],dtype=float)

tree = DecisionTreeClassifier(
    max_depth=1,            # keep small → very interpretable
    min_samples_leaf=5,     # prevent overfitting on tiny leaves
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,       # number of trees – default 100 is fine
    max_depth=1,            # limit depth to prevent overfit (optional)
    min_samples_leaf=2,     # prevent tiny leaves
    random_state=42,
    n_jobs=-1               # use all CPU cores for speed
)

gb = GradientBoostingClassifier(
    n_estimators=100,       # number of boosting stages / trees
    learning_rate=0.1,      # step size – smaller = slower but often better
    max_depth=1,            # depth of each tree – keep shallow
    min_samples_leaf=2,     # prevent overfitting
    random_state=42
)

rf.fit(dataset, labels)
tree.fit(dataset, labels)
gb.fit(dataset, labels)
