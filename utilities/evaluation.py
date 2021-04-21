import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def simple_accuracy(predicted_file):
    """calls sklearn for simple accuracy calculation on the given file"""
    
    predicted = pd.read_csv(predicted_file, header=0)
    for col in [col for col in predicted.columns if 'prediction' in col]:
        
        score = accuracy_score(predicted['duration_label'], predicted[col])
        print(f"accuracy for '{col}': {(score*100):.2f}%")