import numpy as np

Weather     = {0: 'Sunny',     1:'Cloudy',              2:'Rainy',            3:'Stormy'}
Temperature = {0: 'Below 10°', 1:'Between 10° and 20°', 2:'Greater than 20°', 3:'Over 35°'}

#Weather, temperature, n° of players
features =  np.array([[1,1,4],
                      [3,2,5],
                      [2,2,2],
                      [2,2,4],
                      [1,3,4],
                      [3,2,6],
                      [3,2,8]])

labels = np.array([1,1,0,1,0,0,1])

#P()
probs = np.array([np.means(labels == 0),(np.mean(labels == 1))])

n_classes  = np.unique(labels)
n_features = features.shape[1]

weather_likelihoods = np.zeros((n_classes.shape[0],n_features))
temp_likelihoods    = np.zeros((n_classes.shape[0],n_features))

def computeLikelyhoods(features: np.array):
    for i in range(n_classes):
        for j in range(n_features):
            weather_likelihoods[i][j] = np.sum(features[labels == i][j])/labels.shape[0]
            temp_likelihoods[i][j]    = 
    return

def computeNB(weather, temp, players):
    score = 0 
    return score

# print(features[labels == 1][0])
print(weather_likelihoods)

# I want to calculate:
#     P(play? | Weather=sunny, Temperature=hot, players=4 )
    
#     Score(Y) = P(Y) * P(x1 | Y) * P(x2 | Y) * P(x3 | Y)
    
# In my example:
# 15 yes
# 30 nos

# P(yes) = 15/45 = 0.333
# P(no) = 30/45 = 0.667

# For class Yes:
# P(Weather=sunny |yes) = number of total examples with sunny / total yes
# 8/15 = 0.533

# P(temperature=Hot |yes) = 9 / 15 = 0.6

# P(numberplayers=4|Yes) = 1/15 = 0.067

# score(yes) = 0.333 * 0.533 * 0.6 * 0.067 = 0.0071

# --------------------------------------





# For class No:
# P(Weather=sunny |no) = number of total examples with sunny / total no
# 7/30 = 0.233

# P(temperature=Hot |no) = 18 / 30 = 0.6

# P(numberplayers=4|no) = 2/30 = 0.067

# score(no) = 0.667 * 0.233 * 0.6 * 0.067 = 0.0062



# Predict: Yes (because it has a higher score)

