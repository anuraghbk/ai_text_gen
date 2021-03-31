# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:41:27 2021

@author: Anurag
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 
from math import expm1
import joblib
import pandas as pd
from flask import Flask, jsonify, request,render_template
from tensorflow import keras

app = Flask(__name__)

model = keras.models.load_model("bin/model.h5")

tokenizer = joblib.load("bin/tokenizer.joblib")

def x(seed_text):
    next_words = 25
    max_sequence_len =17
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
        op= seed_text
    return(op)

@app.route("/",)
def my_form():
    return render_template('from_ex.html')



@app.route("/", methods=["POST","GET"])

def plz():
    text = request.form['u']
    output = x(text)
    return render_template('from_ex.html',output=output)


if __name__ == "__main__":
    app.run(debug=True)
    
