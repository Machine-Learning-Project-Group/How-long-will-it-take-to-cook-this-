import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def simple_accuracy(predicted_file):
    """calls sklearn for simple accuracy calculation on the given file"""
    
    predicted = pd.read_csv(predicted_file, header=0)
    for col in [col for col in predicted.columns if 'prediction' in col]:
        
        score = accuracy_score(predicted['duration_label'], predicted[col])
        print(f"Accuracy for '{col}': {(score*100):.2f}%")


def evaluate(predicted_file):
    """prints precision & recall"""
    
    predicted = pd.read_csv(predicted_file, header=0)
    for col in [col for col in predicted.columns if 'prediction' in col]:
        
        print(f"\nNow analyzing performance of '{col}'\n")

        precision = precision_score(predicted['duration_label'], predicted[col], average=None, zero_division=0)
        recall = recall_score(predicted['duration_label'], predicted[col], average=None, zero_division=0)
        f1 = f1_score(predicted['duration_label'], predicted[col], average=None, zero_division=0)
        
        score = pd.DataFrame({'Precision':precision, "Recall":recall, "F_score":f1}, index=[1,2,3])
        print(score)
        
        matrix = confusion_matrix(predicted['duration_label'], predicted[col])
        matrix = pd.DataFrame(matrix, index=[1,2,3], columns=[1,2,3])
        print("\nConfusion matrix:")
        print(matrix, end='\n\n')