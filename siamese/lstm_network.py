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
        '''
        model = Sequential()
        model.add(Embedding(50000, 100, input_length=30))
        model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(30, 100)))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        '''
        encoder_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(50000, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        ])

        target_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        total_model = tf.keras.Sequential([
            encoder_model,target_model
        ])

        return encoder_model,total_model

    def train(self, x_train, y_train):
        encoder, model = self.buildRNN(x_train)
        print(encoder)
        y_train = tf.keras.utils.to_categorical(y_train,num_classes=3)
        print(y_train)
        model.compile(loss='MSE',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=1)
        op = encoder.predict(x_train[0:10])
        print(op.shape)