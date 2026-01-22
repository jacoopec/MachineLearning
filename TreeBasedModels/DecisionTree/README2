BEACH VOLLEY EXAMPLE

I created this simple dataset to manually build a relatively small decision tree.
At each step, I compute the Gini indexes by hand and visualize the resulting tree.
The goal of this tree is to predict whether I would choose to play beach volleyball or not,
based on three features: the weather, the temperature, and the number of players.


Weather   Temperature     Numberplayers        Play?
Sunny        Hot                 4              Yes
Sunny        Hot                 8              Yes
Sunny        Hot                 3              No
Sunny        Cold                3              No
Sunny        Cold                4              Yes
Sunny        Cold                8              Yes
Sunny        Cold                9              No
Sunny        Hot                 9              No
Windy        Hot                 3              No
Windy        Hot                 4              Yes
Windy        Hot                 5              Yes
Windy        Hot                 8              Yes
Windy        Hot                 9              No
Windy        Cold                4              No
Windy        Cold                5              No
Rainy        Cold                4              No
Rainy        Cold                5              No
Rainy        Hot                 2              No
Rainy        Hot                 4              No
Rainy        Hot                 6              Yes
Rainy        Hot                 7              Yes
Rainy        Hot                 8              Yes
Rainy        Hot                 9              No
Stormy       Hot                 4              No
Stormy       Cold                4              No
Stormy       Hot                 3              No
Stormy       Hot                 5              No

We start by computing the Gini index at the root node based on full dataset class distribution of Play.
From the full dataset of 27 samples we have "Yes": 10 and "No": 17.
Gini index before split = 1 - pyes ^2 -pno^2 = 1 -  (10/27)^2 - (17/27)^2 = 0.466392

When evaluating a split in a decision tree, you're not just measuring how pure one side of the split is — you're assessing the overall impurity of the entire partitioning of the data. This is done through a weighted average of the Gini indexes of both groups.

This root Gini index becomes the baseline for evaluating all possible splits.
When a split is made, the tree calculates:
The weighted Gini index of the child nodes (after the split),
Then subtracts this from the original (parent) Gini to compute the Gini gain:
--------------
FIRST SPLIT
At each step, it is possible to split the tree, with one of the features.


If we would decide to Split the tree by the number of players, since this is a numerical feature
we have to compute gini indexes at each possible threshold. 
The possible  values for this feature are these: [2, 3, 4, 5, 6, 7, 8, 9]
So,  we could try to split in the midpoints:
[2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]

Threshold | Left Size | Right Size | Left Class Proportions       | Right Class Proportions       | Weighted Gini

   2.5    |    1      |    26      | {'No': 1.0}                  | {'No': 0.615, 'Yes': 0.385}   |     0.456
   3.5    |    5      |    22      | {'No': 1.0}                  | {'No': 0.545, 'Yes': 0.455}   |     0.404
   4.5    |    13     |    14      | {'No': 0.769, 'Yes': 0.231}  | {'Yes': 0.5, 'No': 0.5}       |     0.430
   5.5    |    17     |    10      | {'No': 0.765, 'Yes': 0.235}  | {'Yes': 0.6, 'No': 0.4}       |     0.404
   6.5    |    18     |     9      | {'No': 0.722, 'Yes': 0.278}  | {'Yes': 0.556, 'No': 0.444}   |     0.432
   7.5    |    19     |     8      | {'No': 0.684, 'Yes': 0.316}  | {'Yes': 0.5, 'No': 0.5}       |     0.452
   8.5    |    23     |     4      | {'No': 0.565, 'Yes': 0.435}  | {'No': 1.0}                   |     0.419

We get the highest Weighted ginis for 3.5  and  5.5.
The Gini Gain for these thresholds would be : 0.466392 - 0.404 = 0.062
------------------------------------------------------------------------
We try now to split on: Temperature
Temperature == Hot
Left Size: 18        | Right Size: 9
Left Class Proportions:  {'No': 0.556, 'Yes': 0.444}
Right Class Proportions: {'No': 0.778, 'Yes': 0.222}
Gini Left: 0.494      | Gini Right: 0.346
Weighted Gini: 0.444
1 - (8/18)^2 - (10/18)^2  = 0.4938
1 - (2/9)^2 - (7/9)^2  = 0.346
18/27*0.494 + 9/27*0.346 = 0.444

Temperature == Cold
Left Size: 9         | Right Size: 18
Left Class Proportions:  {'No': 0.778, 'Yes': 0.222}
Right Class Proportions: {'No': 0.556, 'Yes': 0.444}
Gini Left: 0.346      | Gini Right: 0.494
Weighted Gini: 0.444

Gini Gain: 0.466392 - 0.444 = 0.022
------------------------------------------------------------------------
Split on: Weather

Weather == Sunny
Left Size: 8         | Right Size: 19
Left Class Proportions:  {'Yes': 0.5, 'No': 0.5}
Right Class Proportions: {'No': 0.684, 'Yes': 0.316}
Gini Left: 0.500      | Gini Right: 0.432
Weighted Gini: 0.452
Gini Gain: 0.466392 - 0.452 = 0.014

Weather == Windy
Left Size: 7         | Right Size: 20
Left Class Proportions:  {'No': 0.571, 'Yes': 0.429}
Right Class Proportions: {'No': 0.650, 'Yes': 0.350}
Gini Left: 0.490      | Gini Right: 0.455
Weighted Gini: 0.464
Gini Gain: 0.466392 - 0.464 = 0.014

Weather == Rainy
Left Size: 8         | Right Size: 19
Left Class Proportions:  {'No': 0.625, 'Yes': 0.375}
Right Class Proportions: {'No': 0.632, 'Yes': 0.368}
Gini Left: 0.469      | Gini Right: 0.465
Weighted Gini: 0.466
Gini Gain: 0.466392 - 0.466 = 0.000

Weather == Stormy
Left Size: 4         | Right Size: 23
Left Class Proportions:  {'No': 1.0}
Right Class Proportions: {'No': 0.565, 'Yes': 0.435}
Gini Left: 0.000      | Gini Right: 0.491
Weighted Gini: 0.419
Gini Gain: 0.466392 - 0.419 = 0.047

----------------------------------------------------

The Gini gains for all features at the root level are:

| Feature                  | Gini Gain         |
| -----------------------  | ----------------- |
| **Numberplayers ≤ 3.5**  | **0.0624** ✅ Best |
| Weather     Stormy       |  0.047            |
| Temperature  (Cold | Hot)|  0.022            |

This means that splitting at Number_of_players ≤ 3.5 gives us the most separated and pure groups for the first branch.
The best feature for the first split is Number_of_players ≤ 5, because it produces the greatest impurity reduction (i.e., highest Gini gain).


------------------------------------------------------------------------
LEFT BRANCH IN THE TREE Numberplayers ≤ 3.5

It is a pure node with gini index = 0. With less than 3.5 players it is not possible to play.
This makes sense since the  minimum number is 4 ( 2 vs 2).

------------------------------------------------------------------------
RIGHT BRANCH IN THE TREE Numberplayers > 3.5

The Gini index for this branch is 0.496
Calculated on 22 samples {No: 54.5%, Yes: 45.5%}

The three features provide these indexes:
Weather == Stormy: Gini = 0.431
Temperature == Hot: Gini = 0.448
Numberplayers > 8.5 Gini : 0.404

The Best Next Split is on the feature 'numberplayers' for the Threshold: 8.5
That provides a gini gain of: 0.092


------------------------------------------------


The Right Sub-Branch (Numberplayers > 8.5) has 4 samples, with 4 No, and so a Gini Index: 0.000
It is a pure Node, no more splits required for this branch.
This makes sense because it is not possible to play beachvolleyball with more than 8 people.

Left Sub-Branch (Numberplayers ≤ 8.5) is made by 18 samples with these {Yes: 55.6%, No: 44.4%}
Weighted Gini After This Split is 0.404
This shows a meaningful reduction in impurity compared to the unsplit branch (Gini ≈ 0.496).

To continue the split on this last branch we use the feature Numberplayers with threshold: 5.5,
Weighted Gini Index: 0.296
Since is the best, compared to: 
Weather == Stormy: Gini = 0.370
Temperature == Hot: Gini = 0.401

------------------------------------------------
We now have:

-The node created from Numberplayers > 5.5 is a pure node with 100% of Yes results.

-The other node for Numberplayers ≤ 5.5 with 12 samples with such proportions: {No: 66.7%,Yes: 33.3%} and a Gini Index: 0.444

The best next split is on Feature: Weather(Sunny) with a weighted Gini Index: 0.267
The other features gave:
Temperature == Hot: Gini = 0.389
Numberplayers ≤ 4.5: Gini = 0.438

------------------------------------------------
Now we get:
-A Sunny Node with 2 samples and Yes: 100%

-A non-Sunny Group with 10 samples and these proportions{No: 80%,Yes: 20%}
Gini Index: 0.320
Weighted Gini After This Split: 0.267

Temperature == Hot: Gini = 0.240
Numberplayers ≤ 4.5: Gini = 0.317

------------------------------------------------

-Windy Group has 4 samples with class Proportions:{Yes: 50%, No: 50%} and a Gini Index of 0.500

-Non-Windy Group is a pure node, made by 6 samples with No: 100%.

For the Windy group (with Numberplayers ≤ 5.5), the best next split is on the feature temperature(Hot).

Weighted Gini Index: 0.267

Other options are: 
Numberplayers ≤ 3.5: Gini = 0.400
Weather == Windy: Gini = 0.480

------------------------------------------------

After splitting the Windy group (Numberplayers ≤ 5.5) by Temperature == Hot, we get:

A "Hot Group" with 3 samples and these class Proportions: {Yes: 66.7%,No: 33.3%} with a Gini Index: 0.444
And a pure Cold Group with 2 samples, both are "No".

Weighted Gini After This Split: 0.267

------------------------------------------------

Best next split is:
Feature: Numberplayers with the threshold 3.5

Weighted Gini Index: 0.000

Other Candidates, with the same Gini: 0.444 are
Weather == Windy
Temperature == Hot

After splitting the Hot group (within Windy and Numberplayers ≤ 5.5) by Numberplayers ≤ 3.5, we achieve:

------------------------------------------------
We finally get 2 pure nodes:

Left Final Group (Numberplayers ≤ 3.5):
This is a pure node, with 1 "no" sample, so Gini Index is 0.000

Right Final Group (Numberplayers > 3.5):
Size: 2 "yes" samples


This is a perfectly pure split.

------------------------------------------------------

We have now evaluated and built out all branches of the decision tree to the point of full classification." 