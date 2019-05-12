import os
import re
import pandas as pd
import numpy as np
import spacy
import pickle
import tensorflow as tf
from collections import Counter
from nltk import word_tokenize
from keras.preprocessing import sequence

from sentence_encoder import *
from kmeans_clusterer import TextSummarizer


MAX_NB_WORDS = 200000
EMBEDDING_DIM = 384
MAX_LENGTH = 30
UNKNOWN = "<UNK>"
PADDING = "<PAD>"

main_vocab = {UNKNOWN:0,PADDING:1}
pattern = "[^0-9a-zA-Z\\s\\?\\.,]"

def loadQuestionsFromTrainDF():
    df = pd.read_csv("train.csv")
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
    #test_ids, test_data = loadQuestionsFromTestDF()

    tokenized_train_data = []
    vocabularies = []
    print(len(q))
    for i in range(0, len(q)):
        try:
            token1 = re.sub(pattern, " ", q[i])
        except UnicodeDecodeError:
            continue
        token1 = word_tokenize(token1.strip().lower())
        tokenized_train_data.append([token1])
        vocabularies.extend(token1)

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

    return tokenized_train_data, labels

def prepareTestSentences(test_data):
    tokenized_test_data = []
    main_vocab = pickle.load(open("main_vocab.pkl","rb"))
    for i in range(0, len(test_data)):
        try:
            token1 = re.sub(pattern, " ", test_data[i])
        except UnicodeDecodeError:
            continue
        token1 = word_tokenize(token1.strip().lower())
        tokenized_test_data.append([token1])

    for i, test_record in enumerate(tokenized_test_data):
        qu1 = test_record[0]
        qu1 = [main_vocab[tok] if tok in main_vocab else main_vocab[UNKNOWN] for tok in qu1]
        tokenized_test_data[i] = [qu1]

    return tokenized_test_data


def getEncoder():

    print("train and tesst")

    train_data, labels = prepareTrainData()


    for i,lbl in enumerate(labels):
            if lbl == 'EAP':
                labels[i] = 0
            elif lbl == 'HPL':
                labels[i] = 1
            else:
                labels[i] = 2


    train_q1 = [rec[0] for rec in train_data]
    print(train_q1)
    train_q1 = sequence.pad_sequences(train_q1, maxlen=MAX_LENGTH)

    print(train_q1.shape)

    train_q1 = np.asarray(train_q1)

    train_question1 = train_q1[0:]
    train_labels = labels[0:]

    print(np.asarray(train_q1).shape)
    vocab_size = len(main_vocab)
    siamese_nn = SentenceEncoder()
    print(len(train_question1))
    print(len(train_labels))
    encoder = siamese_nn.getEncoder(train_question1, train_labels)
    encoder.save('encoder_model.pkl')
    pickle.dump(main_vocab,open("main_vocab.pkl","wb"))


def summarize_text():

    test_text = """Former President Barack Obama was stung and brooding in the aftermath of Hillary Clinton's loss to President Donald Trump, according to new sections of a book released next week, and partly blamed his former secretary of state for the loss.
In new chapters of "Obama: The Call of History," New York Times correspondent Peter Baker depicts the former president questioning his legacy and personally insulted that voters elected Trump instead of Clinton, whom he campaigned for heavily in the closing days of the 2016 contest.
"This stings," Obama told staffers during an Oval Office meeting in the weeks after the election, a source told David Remnick of The New Yorker. "This hurts."
Baker describes Obama as deeply skeptical of Trump and mystified at why voters chose him over the more experienced Clinton. And he writes that Obama and some of his aides faulted Clinton and her campaign decisions for the outcome, which shocked the then-president.
"To Obama and his team, however, the real blame lay squarely with Clinton," Baker writes in the book. "She was the one who could not translate his strong record and healthy economy into a winning message. Never mind that Trump essentially ran the same playbook against Clinton that Obama did eight years earlier, portraying her as a corrupt exemplar of the status quo. She brought many of her troubles on herself."""

    from nltk.tokenize import sent_tokenize
    texts = sent_tokenize(test_text)
    print(texts)
    test_text = prepareTestSentences(texts)
    test_text = [rec[0] for rec in test_text]
    print(test_text)
    test_text = sequence.pad_sequences(test_text, maxlen=MAX_LENGTH)
    print(test_text)

    encoder_model = tf.keras.models.load_model('encoder_model.pkl')
    op = encoder_model.predict(test_text)
    print(op.shape)
    summarizer = TextSummarizer()
    summary = summarizer.summarize(texts,op)
    print(summary)

#getEncoder()
summarize_text()
