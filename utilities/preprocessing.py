import string
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def is_verb(word):
    return 'v' in set(s.pos() for s in wn.synsets(word) if s.name().split('.')[0]==word)

def remove_non_verb(text):
    pos_tagged = nltk.pos_tag(text)
    if pos_tagged:
        verbs = [pos_tagged[0][0]] + [i[0] for i in pos_tagged[1:] if 'VB' in i[1]] # always include 1st word as nltk doesnt perform well here
        return verbs

def preprocess(file, output=None, rule=1):
    
    data = pd.read_csv(file, header=0)

    # convert dtype to list of string
    # split text sections by comma
    data['processed_steps'] = data['steps'].apply(lambda x: x[1:-1].split(","))
    
    # tokenize
    data['processed_steps'] = data['processed_steps'].apply(lambda steps: [nltk.word_tokenize(step) for step in steps])
    
    # drop non-verb words by pos-tag
    data['processed_steps'] = data['processed_steps'].apply(lambda steps: [remove_non_verb(step) for step in steps if remove_non_verb(step)])
    
    # stopword removal
    # lemmatization
    # + flatten
    data['processed_steps'] = data['processed_steps'].apply(lambda steps: [lemmatizer.lemmatize(word, pos='v') for step in steps for word in step if word not in stop_words])
    
    # remove punctuation
    data['processed_steps'] = data['processed_steps'].apply(lambda steps: [word.translate(str.maketrans('', '', string.punctuation)) for word in steps])
    
    # drop by synsets
    data['processed_steps'] = data['processed_steps'].apply(lambda steps: [word for word in steps if is_verb(word)])
    
    # counting verbs
    data['n_verbs'] = data['processed_steps'].apply(lambda steps: len(set(steps)))
    
    a = list(data['processed_steps'])
    verbs = [j for i in a for j in i]
    print(f"Steps processed into {len(verbs)} verbs, containing {len(set(verbs))} unique verbs")
    
    if output:
        data.to_csv(output, header=True, index=False)
        print(f"Processed {file}, saved to {output}\n")
    else:
        data.to_csv(file, header=True, index=False)
        print(f"Processed {file}\n")