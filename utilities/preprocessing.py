import string
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def is_verb(word):
    """check if the given word is a verb using synonym set provided by nltk"""
    return 'v' in set(s.pos() for s in wn.synsets(word) if s.name().split('.')[0]==word)

def convert_list(obj):
    """-Removed-"""
    return [i for i in re.findall("[\w\s]+", obj) if any([j.isalnum() for j in i.split()])]

def extract_verb(file, output=None):
    """
    extract verbs from steps,
    file: csv for processing,
    output(optional): output csv saving path
    """
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