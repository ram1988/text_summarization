# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *



# https://towardsdatascience.com/first-contact-with-tensorflow-estimator-69a5e072998d
# https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/RNNEstimator
class SentenceEncoder:

    def buildRNN(self, x):
        print(x)
        # bidirectional rnn to get the sentence vectors
        # https://blog.myyellowroad.com/unsupervised-sentence-representation-with-deep-learning-104b90079a93
        # https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a
        model = Sequential()
        model.add(Embedding(50000, 100, input_length=50))
        model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
        model.add(Dense(5))
        model.add(Activation('softmax'))

        return model

    def train(self, x_train, y_train):
        model = self.buildRNN(x_train)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5)