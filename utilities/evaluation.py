import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
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
    
def proba_2_pred(probas):
    """
    Convert dataframe of probabilities into dataframe of prediction
    probas: dataframe in the form [model 1 label 1 probability], [model 2 label 2 probability], ... [model n label n probability]
    """
    length = probas.values.shape[1]
    out = pd.DataFrame(range(probas.shape[0]))

    for i in range(0, length, 3):
        proba = X_t.iloc[:, i:i+3]
        yhat = pd.Series(np.argmax(proba.values, axis=1)+1, name=f'{proba.columns[0][:-2]}')
        out = pd.concat([out, yhat], axis=1)

    out.drop(0, axis=1, inplace=True)
    return out

def shared_error(pred_a, pred_b, titles):
    """
    Visualize shared error between 2 predictions,
    pred_a: a series of boolean value indicating model a performance
    """
    print(f"Model {titles[0]} accuracy : {pred_a.sum()/len(pred_a.index)}")
    print(f"Model {titles[1]} accuracy : {pred_b.sum()/len(pred_b.index)}")

    TT = TF = FF = 0

    length = len(pred_a.index)

    for i in range(length):
      a = pred_a.iloc[i]
      b = pred_b.iloc[i]
      if a and b:
          TT += 1
      elif not (a or b):
          FF += 1
      else:
          TF += 1

    print(f"Out of all predictions made,")

    # Pie chart
    labels = 'Shared error', 'Theoritical improvement potential', 'Shared correct prediction'
    sizes = [FF, TF, TT]
    explode = (0.1, 0.1, 0.1)  # only "explode" the 2nd slice
    colors = ['tomato', 'cyan', 'springgreen']

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, colors=colors, labels=labels, autopct='%1.1f%%', startangle=120)
    ax1.axis('equal')

    plt.show()
    
def evaluate_models(yhats, y, titles):
    """
    Evaluate performance of multiple models
    
    yhats: 2d array of predictions
    y: array of true label
    titles: list of model names
    """
    for yhat, title in zip(yhats, titles):
        print(f"Evaluating {title}:")
        evaluate(yhat, y)
        print("-"*50)
    
    if len(yhats) > 1:
        length = len(yhats)
        for comb in combinations(range(length), 2):
          a = comb[0]
          b = comb[1]
          names = [titles[a], titles[b]]
          print(f"Evaluating {titles[a]} & {titles[b]}")
          a = yhats[a]
          b = yhats[b]

          pred_a = pd.Series(a==y)
          pred_b = pd.Series(b==y)
          print("passing")
          shared_error(pred_a, pred_b, names)