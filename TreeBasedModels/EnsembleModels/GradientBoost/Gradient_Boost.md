Decision trees:

Handle non-linear relationships
Handle feature interactions automatically
Require no feature scaling
Can fit complex residual patterns

Gradient Boosting builds trees sequentially, where each new tree learns to correct the mistakes of the current model by fitting the negative gradient of the loss function.

Gradient boosting builds a strong predictor by iteratively adding small decision trees that follow the gradient of the loss function to correct previous errors.

Update the model
Add the new tree to the model:
Fm(x) = Fm-1(x) + n * hm(x)

Why “gradient” boosting?
Because:
The residuals are gradients of the loss
The algorithm performs gradient descent in function space
Trees are the basis functions