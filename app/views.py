from django.shortcuts import render
from .cleaning import cleaning

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Bidirectional, SpatialDropout1D
from keras.models import load_model

def index(request):
    if request.method == 'POST':
        df_train = pd.read_csv('app/Corona_NLP_train.csv', encoding='latin_1')
        df_train['OriginalTweet'] = df_train['OriginalTweet'].apply(lambda x: cleaning(x))

        encoding = {
            'Extremely Negative':0,
            'Negative':0,
            'Neutral':1,
            'Positive':2,
            'Extremely Positive':2
        }
        df_train['Sentiment'] = df_train.Sentiment.replace(encoding, inplace=True)

        text = str(request.POST['tweet'])
        text = np.array([text])
        text = pd.Series(text)
        text = text.apply(cleaning)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df_train['OriginalTweet'])

        text = tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, padding='post')
        
        model = load_model('app/model_cnn.h5')
        # print(model.predict_classes(text))
        context = {
            'title':'Sentiments Detector',
            'text':str(request.POST['tweet']),
            'result':model.predict_classes(text)
        }
    else:
        context = {
            'title':'Sentiments Detector'
        }
    return render(request, 'app/index.html', context)