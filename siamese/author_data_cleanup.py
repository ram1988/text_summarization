import os
import re
import pandas as pd
import numpy as np
import spacy
import pickle
from collections import Counter
from nltk import word_tokenize
from siamese.lstm_network import *

MAX_NB_WORDS = 200000
EMBEDDING_DIM = 384
MAX_LENGTH = 30
UNKNOWN = "<UNK>"
PADDING = "<PAD>"

main_vocab = {UNKNOWN:0,PADDING:1}

def loadQuestionsFromTrainDF():
    df = pd.read_csv("../train.csv")
    return df["text"],df["author"]

def loadQuestionsFromTestDF():
    df = pd.read_csv("../data/test.csv")
    return df["id"],df["text"]


def __prepareEmbeddingMatrix(vocabulary):
    nlp = spacy.load('en')
    embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))

    for word, idx in vocabulary.items():
        doc = nlp(word)
        vec = doc.vector
        embedding_matrix[idx] = vec

    return embedding_matrix

def prepareTrainData():

    q, labels = loadQuestionsFromTrainDF()
    test_ids, test_data = loadQuestionsFromTestDF()

    tokenized_train_data = []
    vocabularies = []
    pattern = "[^0-9a-zA-Z\\s\\?\\.,]"
    print(len(q))
    for i in range(0, len(q)):
        try:
            token1 = re.sub(pattern, " ", q[i])
        except UnicodeDecodeError:
            continue
        token1 = word_tokenize(token1.strip().lower())
        tokenized_train_data.append([token1])
        vocabularies.extend(token1)

    tokenized_test_data = []
    for i in range(0, len(test_data)):
        try:
            token1 = re.sub(pattern, " ", test_data[i])
        except UnicodeDecodeError:
            continue
        token1 = word_tokenize(token1.strip().lower())
        tokenized_test_data.append([token1])


    vocabCounter = Counter(vocabularies).most_common()
    idx = len(main_vocab)
    for i in vocabCounter:
        if len(main_vocab) < MAX_NB_WORDS:
            main_vocab[i[0]] = idx
            idx = idx + 1

    print(len(main_vocab))

    print(main_vocab)

    for i, train_record in enumerate(tokenized_train_data):
        qu1 = train_record[0]
        qu1 = [main_vocab[tok] if tok in main_vocab else main_vocab[UNKNOWN] for tok in qu1]
        tokenized_train_data[i] = [qu1]

    for i, test_record in enumerate(tokenized_test_data):
        qu1 = test_record[0]
        qu1 = [main_vocab[tok] if tok in main_vocab else main_vocab[UNKNOWN] for tok in qu1]
        tokenized_test_data[i] = [qu1]

    print("prep test data")
    embedding_matrix = __prepareEmbeddingMatrix(main_vocab)
    final_input = (tokenized_train_data, (test_ids,tokenized_test_data) , labels, embedding_matrix)
    pickle.dump(final_input, open("traindata.pkl", "wb"))
    pickle.dump(main_vocab, open("main_vocab.pkl", "wb"))

    return tokenized_train_data, (test_ids,tokenized_test_data), labels, embedding_matrix


def runModelWithEmbed():
    from keras.preprocessing import sequence

    print("train and tesst")

    train_data, test_data, labels, embedding_matrix = prepareTrainData()


    for i,lbl in enumerate(labels):
            if lbl == 'EAP':
                labels[i] = 0
            elif lbl == 'HPL':
                labels[i] = 1
            else:
                labels[i] = 2


    print(labels)

    train_q1 = [rec[0] for rec in train_data]
    train_q1 = sequence.pad_sequences(train_q1, maxlen=MAX_LENGTH)

    print(train_q1.shape)

    train_no = 18000
    train_q1 = np.asarray(train_q1)

    train_question1 = train_q1[0:]
    train_labels = labels[0:]

    validate_question1 = train_q1[train_no:]
    validate_labels = labels[train_no:]

    print(np.asarray(train_q1).shape)
    vocab_size = len(main_vocab)
    siamese_nn = SentenceEncoder()
    siamese_nn.train(train_question1, train_labels)


runModelWithEmbed()
