Fit a regression tree with:
One feature: X
Target: Y
Max depth = 1 → Only 1 split, producing 2 leaf nodes

Since we can only make 1 split, we’ll test all possible cut points between adjacent X values and find the one that minimizes the total squared error (variance).
We'll try splitting at the midpoint between each pair:

Split = X[i] + X[i+1]/2

MSE = sommatoria(yi-yleft)^2 + sommatoria(yi-yright)^2


✅ Step-by-Step Result (Depth = 1)
🔪 Best split point: X = 1.075
🟦 Left region (X ≤ 1.075):
Predict value = 1.31 (mean of Y in this region)
🟥 Right region (X > 1.075):
Predict value = 13.999 (mean of Y in this region)
📉 Total squared error: ≈ 754.92

Your regression tree with depth = 1 learns:
if X <= 1.075:
    predict 1.31
else:
    predict 13.999


EXAMPLE

🔹 Split 1: X = 1.075 (Between 0.95 and 1.2)
➤ Left region (X ≤ 1.075):
X: [0.3, 0.6, 0.65, 0.8, 0.95]
Y: [0.9, 1.2, 1.15, 1.3, 2.0]

Mean(left) = (0.9+1.2+1.15+1.3+2.0)/5 = 1.31
SSE(left) = (0.9-1.31)^2+(1.2-1.31)^2...  = 0.6825

Right region (X > 1.075):
sum of Y = 195.98

Meanright = 195.98 / 14 = 14
SSE(right) = sommatoria(yi-14)^2=754.24
Total SSE = 0.68 + 754.24 =   754.92