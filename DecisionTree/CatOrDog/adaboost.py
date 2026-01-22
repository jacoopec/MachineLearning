import math

data  = {
    "Sunny":          [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    "Windy":          [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
    "Rainy":          [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
    "HotTemperature": [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
}

target={
    "Play?":          [0,1,0,0,1,1,0,1,1,1,1,1,0,1,0,1]
}



samples = 16
out_dictionary = {}

target["weights"] = [1/samples,1/samples,1/samples,1/samples,1/samples,1/samples,1/samples,1/samples,
                   1/samples,1/samples,1/samples,1/samples,1/samples,1/samples,1/samples,1/samples]


tot_err = 0
incorr_indexs = []

for key in data.keys():
    pos_correct_class = 0
    pos_n_corr_class = 0
    neg_n_correct_class  = 0
    neg_correct_class  = 0
    gini_impurityP = 0
    gini_impurityN = 0
    
    out_dictionary[key] = []
    
    for i in range(samples):
        sunny = data[key][i]

        if(sunny == 1):
            if(target["Play?"][i] == 0):
                tot_err = tot_err + target["weights"][i]
                incorr_indexs.append(i)
        else:
            if(target["Play?"][i] == 1):
                tot_err = tot_err + target["weights"][i]
                incorr_indexs.append(i)
                
    amount_of_say = 0.5 * math.log((1-tot_err)/tot_err)
    
    for i in incorr_indexs:
        target["weights"][i] = target["weights"][i] * math.exp(-amount_of_say)
    # print(key)
    # print(tot_err)
    tot_err = 0
    
print(target)
