import random
import json
import pickle
import numpy as np
import re
import emoji
import string

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
import os

script_dir = os.path.dirname(__file__)

def get_json_data(file):
    return json.loads(open(os.path.join(script_dir, file)).read())

lemmetizer = WordNetLemmatizer()

intents = get_json_data('basic_intents.json')
training_intents = get_json_data('training_intents.json')

intents.extend(training_intents)

words = pickle.load(open(os.path.join(script_dir, 'all_words.pickle'), 'rb'))

tags = sorted(list(set([intent['tag'] for intent in intents])))
model = load_model(os.path.join(script_dir, 'chatbot_model.h5'))

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = emoji.demojize(text)
    text = re.sub(r"\:(.*?)\:", '', text)
    text = re.sub(r"\s+", " ", text)
    text = ''.join([lemmetizer.lemmatize(word) for word in text])
    return text

def bag_of_words(sentence: str):
    cleaned_sentence = clean_text(sentence)
    sentence_words = nltk.word_tokenize(cleaned_sentence)
    
    bag = []

    for word in words:
        bag.append(1) if word in sentence_words else bag.append(0)
    
    return np.array(bag)

def predict(sentence) -> str:
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))
    ERR_TRESHOLD = 0.25

    highest_probability = max(res[0])

    if highest_probability < ERR_TRESHOLD:
        return None
    
    index = np.where(res == highest_probability)[-1][0]

    return tags[index]

def get_response(tag: str) -> str:
    learning_sentence = "I don't know the answer. I am still learning"

    for i in intents:
        if i['tag'] == tag:
            res = i['responses']
    
    if tag == None or res == None:
        return learning_sentence
    else:
        return random.choice(res)

if __name__ == '__main__':
    print("GO! BOT IS RUNNING!")

    while True:
        message = input('')
        tag = predict(message)
        res = get_response(tag)
        print(res)