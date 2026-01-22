data = {
    "Sunny":          [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    "Windy":          [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
    "Rainy":          [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
    "HotTemperature": [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
}

target={
    "Play?":          [0,1,0,0,1,1,0,1,1,1,1,1,0,1,0,1]
}

sample_length = 16
print("data.keys()")
print(data.keys())
out_dictionary = {}

for key in data.keys():
    nOfYes_pos_feat = 0
    nOfNo_pos_feat = 0
    nOfYes_neg_feat  = 0
    nOfNo_neg_feat  = 0
    gini_impurityP = 0
    gini_impurityN = 0
    
    out_dictionary[key] = []
    
    for i in range(sample_length):
        sunny = data[key][i]

        if(sunny == 1):
            if(target["Play?"][i] == 1):
                nOfYes_pos_feat = nOfYes_pos_feat + 1
            else :
                nOfNo_pos_feat = nOfNo_pos_feat + 1
        else:
            if(target["Play?"][i] == 1):
                nOfYes_neg_feat = nOfYes_neg_feat + 1
            else :
                nOfNo_neg_feat = nOfNo_neg_feat + 1
    
    gini_impurityN = 1 - (nOfYes_neg_feat / (nOfNo_neg_feat + nOfYes_neg_feat))**2 - (nOfNo_neg_feat / (nOfNo_neg_feat + nOfYes_neg_feat))**2  
    gini_impurityP = 1 - (nOfYes_pos_feat / (nOfNo_pos_feat + nOfYes_pos_feat))**2 - (nOfNo_pos_feat / (nOfNo_pos_feat + nOfYes_pos_feat))**2 
    
    tot_gini = (nOfYes_pos_feat + nOfNo_pos_feat) / ( nOfYes_pos_feat + nOfNo_pos_feat)*gini_impurityP + (nOfYes_pos_feat + nOfNo_pos_feat) / ( nOfYes_pos_feat + nOfNo_pos_feat + nOfNo_neg_feat +nOfYes_neg_feat)*gini_impurityN
    out_dictionary[key].append(tot_gini)
    
print(out_dictionary)    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        