import numpy as np
import pandas as pd

def n_1R(train_file, rule="steps"):
    """Trains on the given csv file, based on the given rule (steps or ingredients)
    Returns the learned 1R model in the form of a dataframe,
    rule_value: most possible label"""
    
    train = pd.read_csv(train_file, header=0)

    if rule == "steps":
        X = train['n_steps']
    elif rule == "ingredients":
        X = train['n_ingredients']
    elif rule == "verbs":
        X = train['n_verbs']
    else:
        print("Invalid rule! Please choose from ['steps', 'ingredients', 'verbs']")
        return 0
    
    print(f"training on {train_file} using rule: '{rule}'")
    
    y = train['duration_label']
    
    # count frequency of each n_steps in each duration label
    matrix = pd.DataFrame(index=set(X), columns=[1, 2, 3]).fillna(0)
    for i in range(len(X)):
        matrix.loc[X[i], y[i]] += 1
    
    # calculate the most frequent label in each n_step, save to list
    predict = []
    for index, a in matrix.iterrows():
        predict.append([i for i, j in enumerate(a) if j == max(a)][0] + 1)
    
    # save prediction & return
    matrix['predict'] = predict
    return matrix['predict']


def predict_1R(test_file, model, rule='steps', name='1R_prediction'):
    """Predicts the labels from the given file & 1R model"""
    test = pd.read_csv(test_file, header=0)
    
    prediction = []
    for index, row in test.iterrows():
        
        if rule == 'steps':
            X = row['n_steps']
        elif rule == 'ingredients':
            X = row['n_ingredients']
        elif rule == 'verbs':
            X = row['n_verbs']
        
        if X in model.index:
            prediction.append(model[X])
        
        else:
            prediction.append(2)
    
    test[name] = prediction
    print(f"predicted {test_file}, saved to column '{name}'")
    test.to_csv(test_file, index=False)