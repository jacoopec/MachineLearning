I want to calculate:
    P(play? | Weather=sunny, Temperature=hot, players=4 )
    
    Score(Y) = P(Y) * P(x1 | Y) * P(x2 | Y) * P(x3 | Y)
    
In my example:
15 yes
30 nos

P(yes) = 15/45 = 0.333
P(no) = 30/45 = 0.667

For class Yes:
P(Weather=sunny |yes) = number of total examples with sunny / total yes
8/15 = 0.533

P(temperature=Hot |yes) = 9 / 15 = 0.6

P(numberplayers=4|Yes) = 1/15 = 0.067

score(yes) = 0.333 * 0.533 * 0.6 * 0.067 = 0.0071

--------------------------------------

For class No:
P(Weather=sunny |no) = number of total examples with sunny / total no
7/30 = 0.233

P(temperature=Hot |no) = 18 / 30 = 0.6

P(numberplayers=4|no) = 2/30 = 0.067

score(no) = 0.667 * 0.233 * 0.6 * 0.067 = 0.0062



Predict: Yes (because it has a higher score)

