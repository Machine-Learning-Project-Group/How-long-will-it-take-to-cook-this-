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

<<<<<<< Updated upstream
def remove_non_verb(text):
    pos_tagged = nltk.pos_tag(text)
    if pos_tagged:
        verbs = [pos_tagged[0][0]] + [i[0] for i in pos_tagged[1:] if 'VB' in i[1]] # always include 1st word as nltk doesnt perform well here
        return verbs

=======
>>>>>>> Stashed changes
def convert_list(obj):
    """-Removed-"""
    return [i for i in re.findall("[\w\s]+", obj) if any([j.isalnum() for j in i.split()])]

<<<<<<< Updated upstream
def extract_verb(file, output=None, rule=1):
    
    print(f"Processing {file} ...")
    
    data = pd.read_csv(file, header=0)

    # convert dtype to list of string
    # split text sections by comma
    data['processed_steps'] = data['steps'].apply(convert_list)
    
    # tokenize
    data['processed_steps'] = data['processed_steps'].apply(lambda steps: [nltk.word_tokenize(step) for step in steps])
    
    # drop non-verb words by pos-tag
    data['processed_steps'] = data['processed_steps'].apply(lambda steps: [remove_non_verb(step) for step in steps if remove_non_verb(step)])
    
    # remove punctuation
    # + flatten
    data['processed_steps'] = data['processed_steps'].apply(lambda steps: [word.translate(str.maketrans('', '', string.punctuation)) for step in steps for word in step])

    # lemmatization
    data['processed_steps'] = data['processed_steps'].apply(lambda steps: [lemmatizer.lemmatize(word, pos='v') for word in steps])

    # stopword removal
    # + drop by synsets
    data['processed_steps'] = data['processed_steps'].apply(lambda steps: [word for word in steps if (is_verb(word) and word not in stop_words)])
    
    # counting verbs
    data['n_verbs'] = data['processed_steps'].apply(lambda steps: len(set(steps)))
=======
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
>>>>>>> Stashed changes
    
    a = list(data['verbs'])
    verbs = [j for i in a for j in i]
    print(f"Steps processed into {len(verbs)} verbs, containing {len(set(verbs))} unique verbs")
    
    if output:
        data.to_csv(output, header=True, index=False)
        print(f"Processed {file}, saved to {output}\n")
    else:
        data.to_csv(file, header=True, index=False)
        print(f"Processed {file}\n")