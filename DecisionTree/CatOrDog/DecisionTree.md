How the decision tree algorithm builds the tree from scratch, based on your dataset with categorical features (Weather, Mood) and a categorical label (Drink = Tea or Coffee)?

The algorithm works recursively:
Start with all the data.
Try all features to find the best split — the one that gives the lowest Gini impurity after splitting.
Split the data using that feature.
Repeat the process on each subset, using remaining features.
Stop when:
All examples in a node are the same class → make a leaf.
No features are left → choose the most common class.

Data

Weather	Mood	Drink
Sunny	Happy	Tea
Sunny	Tired	Coffee
Rainy	Happy	Tea
Rainy	Tired	Coffee
Sunny	Tired	Coffee
Rainy	Happy	Tea
Sunny	Happy	Tea
Rainy	Tired	Coffee

splitting by Weather:
Sunny → 4 samples: ['Tea', 'Coffee', 'Coffee', 'Tea']
Rainy → 4 samples: ['Tea', 'Coffee', 'Tea', 'Coffee']
Gini for each subset:
Sunny: 2 Tea, 2 Coffee → Gini = 1 - (0.5² + 0.5²) = 0.5
Rainy: 2 Tea, 2 Coffee → Gini = 0.5
Weighted Gini = 0.5

splitting by Mood:
Happy → 4 samples: ['Tea', 'Tea', 'Tea', 'Tea'] → Gini = 0
Tired → 4 samples: ['Coffee', 'Coffee', 'Coffee', 'Coffee'] → Gini = 0
Weighted Gini = 0
This is better than 0.5 — so we split on Mood first!

make the split:
Mood = Happy → all Tea → leaf
Mood = Tired → all Coffee → leaf

Final tree:
Mood = Happy:
  → Tea
Mood = Tired:
  → Coffee


Summary:
Step	What It Does
1	Try each feature
2	Split dataset by unique values of each feature
3	Calculate Gini impurity of each split
4	Pick the feature with the lowest weighted Gini
5	Recurse on each subset
6	Stop when data is pure or no features remain