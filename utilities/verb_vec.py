import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from sklearn import decomposition

tokenizer = RegexpTokenizer(r'\w+')

def word_count(data, words):
    """
    return word cound matrix,
    data: series of list of words,
    words: list of sorted, unique words extracted from train dataset
    """
    counts = pd.DataFrame(index=data.index, columns=words).fillna(0)
    for index, verbs in data.iteritems():
        for verb in verbs:
            if verb in words:
                counts.loc[index, verb] +=1
    return counts

def verb_vec(train_file, test_file, pca_num=50):
    """
    converts verb list into numberical vectors using word count & PCA,
    pca_num: number of principal components"""
    
    train = pd.read_csv(train_file, header=0)
    test = pd.read_csv(test_file, header=0)
    
    # tokenize
    train['verbs'] = train['verbs'].apply(tokenizer.tokenize)
    test['verbs'] = test['verbs'].apply(tokenizer.tokenize)
    
    # extract unique word list from train
    words = sorted(set([j for i in list(train['verbs']) for j in i]))
    
    # construct word count matrix
    train_count = word_count(train['verbs'], words)
    test_count = word_count(test['verbs'], words)
    
    # apply pca to word count
    pca = decomposition.PCA(n_components=pca_num)
    pca.fit(train_count)
    X_train = pca.transform(train_count)
    X_test = pca.transform(test_count)
    
    # merge to original file
    X_train = pd.DataFrame(X_train, columns=["v_vec_"+str(i+1) for i in range(pca_num)])
    X_test = pd.DataFrame(X_test, columns=["v_vec_"+str(i+1) for i in range(pca_num)])
    
    train = pd.merge(train, X_train, left_index=True, right_index=True)
    test = pd.merge(test, X_test, left_index=True, right_index=True)
    
    print(f"Created {pca_num} verb vectors feaatures")
    
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)