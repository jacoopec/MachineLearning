

samples1 = 9
samples2 = 18
samplesTot = samples2 + samples1

no1 = 7/samples1
yes1 = 2/samples1

gini1 = 1 - yes1 *  yes1 - no1 * no1

yes = 8/samples2
no = 10/samples2

gini2 = 1 - yes *  yes - no * no

giniWeighted = samples1/samplesTot*gini1 + samples2/samplesTot*gini2
print("Gini1 ", gini1, " Gini2 ", gini2)
print("Gini weighted ",giniWeighted)
print("Gini gain ",giniWeighted-0.4959)





# samples1 = 8
# samples2 = 8
# samples3 = 7
# samples4 = 4
# samplesTot = samples2 + samples1 + samples3 +  samples4

# yes1 = 4/samples1
# no1 = 4/samples1

# gini1 = 1 - yes1 *  yes1 - no1 * no1

# yes2 = 3/samples2
# no2 = 5/samples2

# gini2 = 1 - yes2 *  yes2 - no2 * no2

# yes3 = 3/samples3
# no3 = 4/samples3

# gini3 = 1 - yes3 *  yes3 - no3 * no3

# giniWeighted = samples1/samplesTot*gini1 + samples2/samplesTot*gini2 + samples3/samplesTot*gini3
# print("Gini1 ", gini1, " Gini2 ", gini2, " Gini3 ",gini3)
# print("Gini weighted ",giniWeighted)
# print("Gini gain ",0.466392-giniWeighted)






import pandas as pd

df =  pd.read_fwf("beachvolley_data.txt")
print(df)
# df_sorted = df.sort_values(by="Numberplayers", ascending=False)
# filtered_df = df[df["Numberplayers"] >3.5]
# filtered_df1 = filtered_df[filtered_df["Temperature"] == "Hot"]
# filtered_df2 = filtered_df1[filtered_df1["Play?"] == "Yes"]
# df_sorted = filtered_df.sort_values(by=df.columns[2], ascending=False)
# print(filtered_df2)

# print(df_sorted)
yeses = (df["Play?"] == "No").sum()
print(yeses)
# print(df.columns[2])
# print(df.iloc[:, -1].unique())

total_samples = 27
totYeses = 10
totNos  = 17

def computeGini(sampBelowe,yesesBelow, nosBelow ):
    sampAbove = total_samples - sampBelowe
    yesAbove = totYeses - yesesBelow
    noAbove = totNos - nosBelow
    giniAbove = 1 - (yesAbove/sampAbove)**2 - (noAbove/sampAbove)**2
    giniBelow = 1 - (yesesBelow/sampBelowe)**2 - (nosBelow/sampBelowe)**2
    return sampAbove /total_samples * giniAbove + sampBelowe / total_samples * giniBelow

# 2 -> Y: 0 N:1
# 2 -> Y: 0 N:1
print("Gini for  4.5" , 0.466392 - computeGini(8,3,5))
print("Gini for  5.5" , 0.466392 - computeGini(12,4,8))
print("Gini for  6.5" , 0.466392 - computeGini(13,5,8))
print("Gini for  7.5" , 0.466392 - computeGini(14,6,8))
print("Gini for  8.5" , 0.466392 - computeGini(18,10,8))

# Group A: Number_of_players ≤ 5
# Samples: 13
# Class Distribution: 10 No, 3 Yes
# Gini Index: 0.355

# Group B: Number_of_players > 5
# Samples: 8
# Class Distribution: 5 Yes, 3 No
# Gini Index: 0.469

# Weighted Gini After Split:
# 13/21*0.355 + 8/21*0.469 = 0.398