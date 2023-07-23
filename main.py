import keras.models
import tensorflow as tf
import json
import spacy
from flask import Flask, request, json
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from gevent.pywsgi import WSGIServer

from keras.models import Sequential
from keras.layers import Dense

from random import choice

# print(device_lib.list_local_devices())

app = Flask("NetChatBot")
cors = CORS(app, resources={r"/*": {"origins": "*"}})

EPOCHS = 250
NAME_MODEL = "chatbot_v2_" + str(EPOCHS)


def initialize():
    f = open('datasetv2.json', encoding="utf-8")
    return json.load(f), spacy.load("it_core_news_lg"), CountVectorizer(), LabelEncoder()


def train(ds, nlp, bow, le):
    dizionario = set({})
    intents = []
    docs = []

    for intent in ds["intents"]:
        for sample in intent["samples"]:
            sample = sample.lower()
            tokens = nlp(sample)
            doc = ""
            for token in tokens:
                if not token.is_punct and not token.is_stop:
                    doc += " " + str(token.lemma_)
                    dizionario.add(token.lemma_)
            if len(doc) > 0:
                docs.append(doc.rstrip())
                intents.append(intent["name"])
    print("Lunghezza del dizionario: %d" % len(dizionario))

    X = bow.fit_transform(docs)
    X = X.toarray()
    print(X.shape)

    y = le.fit_transform(intents)

    ohe = OneHotEncoder()
    y = ohe.fit_transform(y.reshape(-1, 1))
    y = y.toarray()
    print(y.shape)

    X, y = shuffle(X, y, random_state=0)

    model = Sequential()
    model.add(Dense(16, activation="relu", input_dim=X.shape[1]))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(y.shape[1], activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()

    model.fit(X, y, epochs=EPOCHS)
    model.save(NAME_MODEL)
    print("MODELLO SALVATO")


def get_response(intent_r, ds_r):
    for intent in ds_r["intents"]:
        if intent["name"] == intent_r:
            return choice(intent["responses"])


def preprocess(frase, nlp_p, bow_p, ds_p):
    dizionario = set({})
    intents = []
    docs = []

    for intent in ds_p["intents"]:
        for sample in intent["samples"]:
            sample = sample.lower()
            tokens = nlp_p(sample)
            doc = ""
            for token in tokens:
                if not token.is_punct and not token.is_stop:
                    doc += " " + str(token.lemma_)
                    dizionario.add(token.lemma_)
            if len(doc) > 0:
                docs.append(doc.rstrip())
                intents.append(intent["name"])
    # print("Lunghezza del dizionario: %d" % len(dizionario))

    X = bow_p.fit_transform(docs)
    X = X.toarray()
    # print(X.shape)

    tokens = nlp_p(frase.lower())
    doc = ""
    for token in tokens:
        if not token.is_punct and not token.is_stop:
            doc += " " + str(token.lemma_)
    x = bow_p.transform([doc])
    return x, intents


def chatbot(frase, ds_c, nlp_c, bow_c, model_c, le_c):
    x, intents = preprocess(frase, nlp_c, bow_c, ds_c)
    y_prob = model_c.predict(x)[0]

    y = le_c.fit_transform(intents)

    if y_prob.max() > .5:
        y = y_prob.argmax()
        intent = le_c.inverse_transform([y])
        return get_response(intent, ds_c)
    else:
        return "Non ho capito , riprova"


def chatta(ds, nlp, bow, model_c, le, messaggio):
    frase = messaggio
    response = chatbot(frase, ds, nlp, bow, model_c, le)
    return response


def chatta_locale(ds, nlp, bow, model_c, le):
    frase = ""
    while frase != 'basta':
        frase = input("\n\n\n\nUtente : ")
        response = chatbot(frase, ds, nlp, bow, model_c, le)
        print("\n AI : " + response)
    return response


def load_model():
    return keras.models.load_model(NAME_MODEL)


if __name__ == "__main__":
    ds_m, nlp_m, bow_m, le_m = initialize()
    # train(ds_m, nlp_m, bow_m, le_m)
    model_m = load_model()
    # chatta_locale(ds_m, nlp_m, bow_m, model_m, le_m)


@app.route('/chatta', methods=['POST'])
def chattaService():
    messaggio = request.get_json()['messaggio']
    if messaggio:
        print('Sto pensando...')
        response = chatta(ds_m, nlp_m, bow_m, model_m, le_m, messaggio)
        print(response)
        return {'response': response, 'result': True}
    else:
        return {'response': 'Il messaggio non può essere vuoto', 'result': False}


@app.route('/echo', methods=['GET'])
def ping_test():
    return 'echo ChatBot'


print("ChatBot avviato correttamente")

http_server = WSGIServer(('0.0.0.0', 5050), app)
http_server.serve_forever()
