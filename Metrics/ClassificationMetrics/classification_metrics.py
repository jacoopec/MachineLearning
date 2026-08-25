
def compute_metrics(y_evaluated,y_true):
    
    count_0_0 = 0
    count_1_1 = 0
    count_0_1 = 0
    count_1_0 = 0
    
    for pred, true in zip(y_evaluated, y_true):
        if pred == 0 and true == 0:
            count_0_0 += 1
        elif pred == 1 and true == 1:
            count_1_1 += 1
        elif pred == 0 and true == 1:
            count_0_1 += 1
        elif pred == 1 and true == 0:
            count_1_0 += 1
    
    accuracy = (count_0_0 + count_1_1) / (count_0_0 + count_1_1 + count_0_1 + count_1_0)
    precision = count_1_1 / (count_1_0 + count_1_1)
    recall = count_1_1 / (count_0_1 + count_1_1)
    F1  = 2  * (precision * recall) / (precision +  recall)
    
    return [accuracy,precision, recall, F1]

y_evaluated = [0,1,0,1,0,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,1]
y_true      = [0,1,0,1,0,0,0,1,0,0,1,1,1,0,0,0,1,1,1,1,1,1]

y_evaluated1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1]
y_true1      = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

print("Accuracy, precision,recall, F1",compute_metrics(y_evaluated1,y_true1))