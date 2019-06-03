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

def loadSentences():
    df = pd.read_csv("train.tsv",sep="\t")
    df["labels"] = 1
    return df["First"],df["labels"]



def __prepareEmbeddingMatrix(vocabulary):
    nlp = spacy.load('en')
    embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))

    for word, idx in vocabulary.items():
        doc = nlp(word)
        vec = doc.vector
        embedding_matrix[idx] = vec

    return embedding_matrix

def prepareTrainData():

    q, labels = loadSentences()
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
    train_q1 = sequence.pad_sequences(train_q1, maxlen=MAX_LENGTH)

    print(train_q1.shape)

    train_q1 = np.asarray(train_q1)

    train_question1 = train_q1[0:450000]
    train_labels = labels[0:450000]

    print(np.asarray(train_q1).shape)
    vocab_size = len(main_vocab)
    siamese_nn = SentenceEncoder()
    print(len(train_question1))
    print(len(train_labels))
    encoder = siamese_nn.getEncoder(train_question1, train_labels)
    encoder.save('encoder_model.pkl')
    pickle.dump(main_vocab,open("main_vocab.pkl","wb"))


def summarize_text():

    test_text = """A software development process is concerned primarily with the production aspect of software development, as opposed to the technical aspect, such as software tools. These processes exist primarily for supporting the management of software development, and are generally skewed toward addressing business concerns. Many software development processes can be run in a similar way to general project management processes. Examples are:

Interpersonal communication and conflict management and resolution. Active, frequent and honest communication is the most important factor in increasing the likelihood of project success and mitigating problematic projects. The development team should seek end-user involvement and encourage user input in the development process. Not having users involved can lead to misinterpretation of requirements, insensitivity to changing customer needs, and unrealistic expectations on the part of the client. Software developers, users, project managers, customers and project sponsors need to communicate regularly and frequently. The information gained from these discussions allows the project team to analyze the strengths, weaknesses, opportunities and threats (SWOT) and to act on that information to benefit from opportunities and to minimize threats. Even bad news may be good if it is communicated relatively early, because problems can be mitigated if they are not discovered too late. For example, casual conversation with users, team members, and other stakeholders may often surface potential problems sooner than formal meetings. All communications need to be intellectually honest and authentic, and regular, frequent, high quality criticism of development work is necessary, as long as it is provided in a calm, respectful, constructive, non-accusatory, non-angry fashion. Frequent casual communications between developers and end-users, and between project managers and clients, are necessary to keep the project relevant, useful and effective for the end-users, and within the bounds of what can be completed. Effective interpersonal communication and conflict management and resolution are the key to software project management. No methodology or process improvement strategy can overcome serious problems in communication or mismanagement of interpersonal conflict. Moreover, outcomes associated with such methodologies and process improvement strategies are enhanced with better communication. The communication must focus on whether the team understands the project charter and whether the team is making progress towards that goal. End-users, software developers and project managers must frequently ask the elementary, simple questions that help identify problems before they fester into near-disasters. While end-user participation, effective communication and teamwork are not sufficient, they are necessary to ensure a good outcome, and their absence will almost surely lead to a bad outcome.[3][4][5]
Risk management is the process of measuring or assessing risk and then developing strategies to manage the risk. In general, the strategies employed include transferring the risk to another party, avoiding the risk, reducing the negative effect of the risk, and accepting some or all of the consequences of a particular risk. Risk management in software project management begins with the business case for starting the project, which includes a cost-benefit analysis as well as a list of fallback options for project failure, called a contingency plan.
A subset of risk management is Opportunity Management, which means the same thing, except that the potential risk outcome will have a positive, rather than a negative impact. Though theoretically handled in the same way, using the term "opportunity" rather than the somewhat negative term "risk" helps to keep a team focused on possible positive outcomes of any given risk register in their projects, such as spin-off projects, windfalls, and free extra resources.
Requirements management is the process of identifying, eliciting, documenting, analyzing, tracing, prioritizing and agreeing on requirements and then controlling change and communicating to relevant stakeholders. New or altered computer system[1] Requirements management, which includes Requirements analysis, is an important part of the software engineering process; whereby business analysts or software developers identify the needs or requirements of a client; having identified these requirements they are then in a position to design a solution.
Change management is the process of identifying, documenting, analyzing, prioritizing and agreeing on changes to scope (project management) and then controlling changes and communicating to relevant stakeholders. Change impact analysis of new or altered scope, which includes Requirements analysis at the change level, is an important part of the software engineering process; whereby business analysts or software developers identify the altered needs or requirements of a client; having identified these requirements they are then in a position to re-design or modify a solution. Theoretically, each change can impact the timeline and budget of a software project, and therefore by definition must include risk-benefit analysis before approval.
Software configuration management is the process of identifying, and documenting the scope itself, which is the software product underway, including all sub-products and changes and enabling communication of these to relevant stakeholders. In general, the processes employed include version control, naming convention (programming), and software archival agreements.
Release management is the process of identifying, documenting, prioritizing and agreeing on releases of software and then controlling the release schedule and communicating to relevant stakeholders. Most software projects have access to three software environments to which software can be released; Development, Test, and Production. In very large projects, where distributed teams need to integrate their work before releasing to users, there will often be more environments for testing, called unit testing, system testing, or integration testing, before release to User acceptance testing (UAT).
A subset of release management that is gaining attention is Data Management, as obviously the users can only test based on data that they know, and "real" data is only in the software environment called "production". In order to test their work, programmers must therefore also often create "dummy data" or "data stubs". Traditionally, older versions of a production system were once used for this purpose, but as companies rely more and more on outside contributors for software development, company data may not be released to development teams. In complex environments, datasets may be created that are then migrated across test environments according to a test release schedule, much like the overall software release schedule.
"""
    top_n = 3
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
    summary = summarizer.summarize(texts,op,top_n)
    print(summary)

#getEncoder()
summarize_text()
