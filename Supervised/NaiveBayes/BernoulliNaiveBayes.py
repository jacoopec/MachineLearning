import numpy as np

# Since the features were binary:  free ∈ {0,1}, meeting ∈ {0,1}
# I use Bernoulli Naive Bayes

def isIt(arg):
    return [arg[0] == 1, arg[1]==1,arg[2]==1,arg[3]==1]

# important | beautiful | done | missing
a = np.array([[0,1,0,0],
              [1,0,1,1],
              [1,1,1,0],
              [0,0,0,0],
              [1,1,1,1],
              [0,0,0,1],
              [1,0,0,0],
              [1,1,0,1],
              [0,1,1,0],
              [1,1,0,0]])
b = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])

clss  = [np.count_nonzero(b == 0), np.count_nonzero(b == 1)]
feats = np.array([[np.count_nonzero(a[:,0]==1)/clss[0],np.count_nonzero(a[:,1]==1)/clss[0],np.count_nonzero(a[:,2]==1)/clss[0],np.count_nonzero(a[:,3]==1)/clss[0]],
                  [np.count_nonzero(a[:,0]==0)/clss[1],np.count_nonzero(a[:,1]==0)/clss[1],np.count_nonzero(a[:,2]==0)/clss[1],np.count_nonzero(a[:,3]==0)/clss[1]]
                  ])
c = np.array([1,1,1,1])
res = isIt(c)
print(f"Is important ?" + str(res[0]) + " is beautiful? " + str(res[1]) + " is  done? " + str(res[2] )+ " is missing? " + str(res[3]))

print(feats)
print(feats[0][c[0]] * feats[0][c[1]] * feats[0][c[2]] * feats[0][c[3]])
print(feats[1][c[0]] * feats[1][c[1]] * feats[1][c[2]] * feats[1][c[3]])