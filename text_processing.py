import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

def tok_and_lem_input(tweet,handles = True, tok = True, lem = True, predict = True):
    input_dict = {0:tweet}
    dataframe = pd.DataFrame(input_dict, index = input_dict.keys())
    tokenizer = TweetTokenizer(strip_handles = handles)
    lemmatizer = WordNetLemmatizer()
    #tokenizing
    dataframe['text_processed'] = [tokenizer.tokenize(row) for row in dataframe[0]]
    #lemmatizing and joining the text back together 
    for row in dataframe.index:
        dataframe['text_processed'].iloc[row] = ' '.join([lemmatizer.lemmatize(word) for word in dataframe['text_processed'][row]])
    #example of what the data looks like
    return dataframe['text_processed']
