import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



y_test = [0,0,1,1,0,1,1,0,0,1,0,0]   #i valori reali, quelli veri
y_pred = [0,0,1,1,0,1,1,0,0,0,1,0]   #i valori predetti
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#Ci sono 6 match tra coppie di 0,  tra predetti e reali (TP)
#Ci sono 4 match tra coppie di 1 
#C?è un caso FN in cui è   stato predetto 1 ma in realtà è 0 
#E un FP in cui è stato predetto 0 ma in realtà è un 1.
#TP  FP
#FN  TN
