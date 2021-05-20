import string
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def is_verb(word):
    """check if the given word is a verb using synonym set provided by nltk"""
    return 'v' in set(s.pos() for s in wn.synsets(word) if s.name().split('.')[0]==word)

def text_preprocess(data, tag='n', output='list'):
    """
    takes a series of string and preprocess it,
    
    tag: pos_tagger when lemmatizing
    
    output options:
    'list': return each row in list of words
    'string': return each row as a single concatenated string
    """
    # replace " to '
    data = data.apply(lambda x: x.replace("\"", "'")) 
    # split
    data = data.apply(lambda x: re.split(r"', '", x)) 
    # tokenize
    data = data.apply(lambda x: [tokenizer.tokenize(i) for i in x])
    # lemmatize
    data = data.apply(lambda x: [[lemmatizer.lemmatize(word, pos=tag) for word in i] for i in x])
    # stopword
    data = data.apply(lambda x: [[word for word in i if word not in stop_words] for i in x])
    # numbers
    data = data.apply(lambda x: [[word for word in i if not word.isnumeric()] for i in x])
    # remove empty lists
    data = data.apply(lambda x: [i for i in x if i])
    
    # formatting
    if output=='list':
        data = data.apply(lambda x: [j for i in x for j in i])
    elif output=='string':
        data = data.apply(lambda x: [' '.join(i) for i in x]).apply(lambda x: ' '.join(x))
    else:
        pass
    
    return data

"""
def extract_verb(file, output=None):

    print(f"Processing {file} ...")
    
    data = pd.read_csv(file, header=0)
    
    # tokenize
    data['verbs'] = data['steps'].apply(lambda steps: tokenizer.tokenize(steps))
    data['tok_ingredients'] = data['ingredients'].apply(lambda ing: tokenizer.tokenize(ing))
    print("\rProgress: + - - - -", end='')
    
    # lemmatization
    data['verbs'] = data['verbs'].apply(lambda steps: [lemmatizer.lemmatize(word, pos='v') for word in steps])
    data['tok_ingredients'] = data['tok_ingredients'].apply(lambda ing: [lemmatizer.lemmatize(i, pos='n') for i in ing])
    print("\rProgress: + + - - -", end='')
    
    # stopword removal
    # + drop by synsets
    data['verbs'] = data['verbs'].apply(lambda steps: sorted([word for word in steps if (is_verb(word) and word not in stop_words)]))
    print("\rProgress: + + + - -", end='')
    
    # remove ingredients
    new = []
    for index, row in data.iterrows():
        new.append([verb for verb in row['verbs'] if verb not in row['tok_ingredients']])
    data['verbs'] = new
    data.drop('tok_ingredients', axis=1, inplace=True)
    print("\rProgress: + + + + -", end='')
    
    # counting verbs
    data['n_verbs'] = data['verbs'].apply(lambda steps: len(set(steps)))
    print("\rProgress: + + + + +")

    a = list(data['verbs'])
    verbs = [j for i in a for j in i]
    print(f"Steps processed into {len(verbs)} verbs, containing {len(set(verbs))} unique verbs")
    
    if output:
        data.to_csv(output, header=True, index=False)
        print(f"Processed {file}, saved to {output}\n")
    else:
        data.to_csv(file, header=True, index=False)
        print(f"Processed {file}\n")
"""

def extract_verb(file, output=None):
    """
    extract verbs from steps,
    file: csv for processing,
    output(optional): output csv saving path
    """
    print(f"Processing {file} ...")
    
    data = pd.read_csv(file, header=0)
    
    data['verbs'] = text_preprocess(data['steps'], tag='v')
    data['tok_ingredients'] = text_preprocess(data['ingredients'], tag='n')
    print("\rProgress: + - - - -", end='')
    
    # drop by synsets
    data['verbs'] = data['verbs'].apply(lambda steps: sorted([word for word in steps if is_verb(word)]))
    print("\rProgress: + + + - -", end='')
    
    # remove ingredients
    new = []
    for index, row in data.iterrows():
        new.append([verb for verb in row['verbs'] if verb not in row['tok_ingredients']])
    data['verbs'] = new
    data.drop('tok_ingredients', axis=1, inplace=True)
    print("\rProgress: + + + + -", end='')
    
    # counting verbs
    data['n_verbs'] = data['verbs'].apply(lambda steps: len(set(steps)))
    print("\rProgress: + + + + +")

    a = list(data['verbs'])
    verbs = [j for i in a for j in i]
    print(f"Steps processed into {len(verbs)} verbs, containing {len(set(verbs))} unique verbs")
    
    if output:
        data.to_csv(output, header=True, index=False)
        print(f"Processed {file}, saved to {output}\n")
    else:
        data.to_csv(file, header=True, index=False)
        print(f"Processed {file}\n")
        
def extract_ingredients(file, output=None):
    
    print(f"Processing {file} ...")
    
    data = pd.read_csv(file, header=0)
    
    # preprocess
    data['ingrs'] = text_preprocess(data['ingredients'], output='makeAmerikaGreatAgain')
    # only keep the last word
    data['ingrs'] = data['ingrs'].apply(lambda x: [i[-1] for i in x])
    
    a = list(data['ingrs'])
    ingredients = [j for i in a for j in i]
    print(f"Ingredients processed into {len(ingredients)} ingrs, containing {len(set(ingredients))} unique ingrs")
    
    if output:
        data.to_csv(output, header=True, index=False)
        print(f"Processed {file}, saved to {output}\n")
    else:
        data.to_csv(file, header=True, index=False)
        print(f"Processed {file}\n")

# ----------------------------n_grams----------------------------
def cv_ize(train_file, test_file, feature='steps', n_grams=(1,2), var_thre=0):
    """
    count vectorize given feature (a series of strings) into n_grams,
    
    feature: 'steps' or 'ingrs',
    n_grams: n gram range for sklearn.countvectorizer,
    var_thre: drop features with < var_thre if non-zero,
    
    return 2 scaled sparse matrix (X, X_test)
    """
    
    # read & preprocess "steps" into a single string
    train = pd.read_csv(train_file, header=0).loc[:, feature]
    test = pd.read_csv(test_file, header=0).loc[:, feature]
    train = text_preprocess(train, output='string')
    test = text_preprocess(test, output='string')
        
    # count vectorize
    cv = CountVectorizer(ngram_range=n_grams)
    X = cv.fit_transform(train)
    X_test = cv.transform(test)
    print(f"Extracted {len(cv.get_feature_names())} grams.")

    # scale
    transformer = MaxAbsScaler()
    X = transformer.fit_transform(X)
    X_test = transformer.transform(X_test)

    # drop var
    if var_thre > 0:
        selector = VarianceThreshold(threshold=var_thre)
        X = selector.fit_transform(X)
        X_test = selector.transform(X_test)
        print(f"Transformed to {X.toarray().shape}.")

    return (X, X_test, cv.get_feature_names())