How gradient boost fits a model to the training data?
When trying to predict a continous value like weight, the first guess is  
the  average value.
like adaboost, this tree is based on the errors made by the previous tree.
Like adaboost, gradient boost builds fixed sizes trees based on the previous 
tree's errors. But unlike adaboost, each tree can be larger than a stump.
Also like adaboost gradient boost scales the trees. but in comparison with adaboost
it scales by the same amount.
then, GB creates another tree based on the errors made by the previous
and then it scales the tree. 
It continous to build trees in this fashion until it has made the number 
of trees you asked for,  or additional trees fail to improve the fit. 
The first prediction is a weighted sum (in regression).
If we would stop now, this mean would be the prediction for all the samples. 
The errors the previous tree made are the differences between the observed weights and the predicted weight, the mean. 
Now, we combine the average computed before with the new tree built on residuals. 
If you now simply add the weight to the residual you get the original values coming from the 
data. 
By scaling the tree you just built with the learning rate, you get predicted values which are  
If multiple samples end up in the same leaf,  we replace them with their averages. 
We add now this new tree to the old one. 
Each time we add a tree to a prediction,  the residual get smaller. 
Then, we keep making trees until we reach a maximum or adding additional  trees 
doesn't reduce the size of the residuals. 
We scale the tree's contribution to the final prediction witha learning rate. 
We keep adding trees based on the errors made  by the previous tree. 
The loss function used when doing Regression with gradient boost is 
    1/1(observed - predicted)^2 

L(y, F(x)) where F(x) is a function that gives us the predicted values. 


GRADIENT BOOST FOR CLASSIFICATION 
The initial prediction is the log of the odds, which can be thought as the 
logistic regression equivalent of the average.

JUst like in logistic regression, the easiest way to use log(odds) 
for classification is to convert it to a probability, we do that with a 
    logistic function. 
        