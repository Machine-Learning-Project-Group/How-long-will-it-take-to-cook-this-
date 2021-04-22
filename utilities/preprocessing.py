import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def is_verb(word):
    return 'v' in set(s.pos() for s in wn.synsets(word) if s.name().split('.')[0]==word)

def preprocess(file):
    
    data = pd.read_csv(file, header=0, index_col='index')

    # convert dtype to list of string
    data['steps'] = data['steps'].apply(lambda x: x[1:-1].split(','))

    # tokenize
    data['steps'] = data['steps'].apply(lambda steps: [tokenizer.tokenize(step) for step in steps])

    # stopword removal
    # + lemmatization
    data['steps'] = data['steps'].apply(lambda steps: [[lemmatizer.lemmatize(word) for word in step if word not in stop_words] for step in steps])

    # drop non-verb words
    data['steps'].apply(lambda steps: [[word for word in step if is_verb(word)] for step in steps])
    
    data.to_csv(file, header=True, index=True)