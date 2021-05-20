import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def evaluate(yhat, y):
    """format print precision, recall, f1 score & confusion matrix"""

    accuracy = accuracy_score(y, yhat)
    print(f"Accuracy = {(accuracy*100):.2f}%")

    precision = precision_score(y, yhat, average=None, zero_division=0)
    recall = recall_score(y, yhat, average=None, zero_division=0)
    f1 = f1_score(y, yhat, average=None, zero_division=0)

    score = pd.DataFrame({'Precision':precision, "Recall":recall, "F_score":f1}, index=[1,2,3])
    print(score)

    matrix = confusion_matrix(y, yhat)
    matrix = pd.DataFrame(matrix, index=[1,2,3], columns=[1,2,3])
    print("\nConfusion matrix:")
    print(matrix, end='\n\n')