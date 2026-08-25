from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

accuracy_list = [0.8,0.82,0.84,0.85]
precision_list = [0.73,0.78,0.8,0.9]
recall_list = [7.9,7.9,8.3,8.4]
f1_list = []

TP = 10
TN = 5
FN = 2
FP = 1


accuracy = (TP + TN)/(TP + TN + FP + FN)
precision = (TP )/(TP + FP) #How many of the predicted positives were actually positive
recall = (TP )/(TP + FN) #How many actual positives were correctly predicted
F1 = 2 * (precision * recall) / (precision + recall)
accuracy_list.append(accuracy)
precision_list.append(precision)
recall_list.append(recall)

for n in range(4):
    F1 = 2 * (precision_list[n] * recall_list[n]) / (precision_list[n]  + recall_list[n])
    f1_list.append(F1)
print ("accuracy " , accuracy)
print ("precision " , precision)
print ("recall " , recall)
print ("F1 " , F1)



plt.figure(figsize=(10,6))
plt.plot(accuracy_list, label='Accuracy')
plt.plot(precision_list, label='Precision')
plt.plot(recall_list, label='Recall')
plt.plot(f1_list, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Classification Metrics Over Epochs')
plt.legend()
plt.grid(True)
plt.show()