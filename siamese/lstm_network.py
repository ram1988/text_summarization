
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn


#Feed the sentences of dynamic length. Based on that, dynamic no of LSTM cells need to be created
#sentence vectors will be generated and fed to the k-means clustering for sentence similarity for summarization.
class LSTMNet:
    start = 0
    model_file = "./author_lstm.model"

    def __init__(self, nfeatures, max_length, vocab_size, embedding_matrix):
        self.nfeatures = nfeatures
        self.n_hidden = nfeatures / 2
        self.n_steps = max_length
        self.n_layers = 1
        self.batch_size = 200
        self.dropout = 0.7
        self.max_length = max_length
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.threshold = 0.7
        self.learning_rate = 0.08
        self.epsilon = 1e-3
        self.istraining = True

    def __createBatch(self, input1=None, labels=None, batch_size=None):

        self.end = self.start + batch_size

        batch_x1 = input1[self.start:self.end]
        print(len(batch_x1))

        batch_y = labels[self.start:self.end]

        self.start = self.end

        if (self.end >= len(input1)):
            self.start = 0

        return batch_x1, batch_y

    def __createTestBatch(self, input1=None, batch_size=None):

        self.end = self.start + batch_size

        batch_x1 = input1[self.start:self.end]
        print(len(batch_x1))

        self.start = self.end

        if (self.end >= len(input1)):
            self.start = 0

        return batch_x1

    def convertLabelsToOneHotVectors(self, labels):

        one_hot_label = []

        for label in labels:
            if label == 0:
                one_hot_label.append([1, 0, 0])
            elif label == 1:
                one_hot_label.append([0, 1, 0])
            else:
                one_hot_label.append([0, 0, 1])

        return one_hot_label

    def reshape(self, input1, labels=None):
        input1 = np.reshape(input1, (-1, self.max_length))
        labels = np.reshape(labels, (-1, 1))

        return input1, labels

    def insertBatchNNLayer(self, mat_rel, axes, dimension_size):
        mean = None
        var = None
        batch_mean, batch_var = tf.nn.moments(mat_rel, axes)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        if self.istraining:
            print("is training in BN")
            mean = batch_mean
            var = batch_var
            ema_apply_op = ema.apply([batch_mean, batch_var])
        else:
            print("is testing in BN")
            mean = ema.average(batch_mean)
            var = ema.average(batch_var)

        scale2 = tf.Variable(tf.ones(dimension_size, dtype=tf.float64), dtype=tf.float64)
        beta2 = tf.Variable(tf.zeros(dimension_size, dtype=tf.float64), dtype=tf.float64)
        bn_layer = tf.nn.batch_normalization(mat_rel, mean, var, beta2, scale2, self.epsilon)

        return bn_layer

    def buildRNN(self, x, scope):
        print(x)
        x = tf.transpose(x, [1, 0, 2])


        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            fw_cell_array = []
            print(tf.get_variable_scope().name)
            for _ in range(self.n_layers):
                fw_cell = rnn.GRUCell(self.n_hidden, activation=tf.nn.relu)
                fw_cell = rnn.DropoutWrapper(fw_cell,input_keep_prob=self.dropout,output_keep_prob=self.dropout)
                fw_cell_array.append(fw_cell)
            fw_cell = rnn.MultiRNNCell(fw_cell_array, state_is_tuple=True)
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            bw_cell_array = []
            print(tf.get_variable_scope().name)
            for _ in range(self.n_layers):
                bw_cell = rnn.GRUCell(self.n_hidden, activation=tf.nn.relu)
                bw_cell = rnn.DropoutWrapper(bw_cell,input_keep_prob=self.dropout,output_keep_prob=self.dropout)
                bw_cell_array.append(bw_cell)
            bw_cell = rnn.MultiRNNCell(bw_cell_array, state_is_tuple=True)


        outputs = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float64, time_major=True)
        print("output-->" + str(outputs))
        outputs = tf.concat(outputs[0], 2)
        outputs = tf.reshape(outputs, [-1, self.nfeatures])
        outputs = tf.split(outputs, self.n_steps, 0)
        print("output-->"+str(outputs))
        outputs = outputs[-1]
        print("output-->" + str(outputs))

        return output

    def optimizeWeights(self, pred):
        #cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(pred), reduction_indices=1))
        print("predicted-->"+str(pred))
        cost = tf.losses.log_loss(self.y, pred)
        #global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
        #                                           1000, 0.5, staircase=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        return optimizer, cost


    def prepareFeatures(self):
        x1 = tf.placeholder(tf.int32, [None, self.max_length])  # batch_size x sentence_length
        y = tf.placeholder(tf.float64, [None, 3], "labels")

        return x1, y


    #connect to fully connected layer of 3 target classes
    def trainModel(self, input1, labels, one_hot_encoding=False):
        # Parameters

        training_epochs = 10
        display_step = 1
        record_size = len(input1)
        labels = self.convertLabelsToOneHotVectors(labels)


        self.x1, self.y = self.prepareFeatures()
        self.embedded_chars1 = tf.nn.embedding_lookup(self.embedding_matrix, self.x1,
                                                      name="lookup1")  # batch_size x sent_length x embedding_size

        print("Embedding-->" + str(self.embedded_chars1))
        print("Embedding-->" + str(self.x1))


        self.pred = self.buildRNN(self.embedded_chars1, "nn1_side")


        # Initializing the variables
        optimizer, cost = self.optimizeWeights(self.pred)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            count = 0
            labels = np.reshape(labels, (-1, 3))

            # Training cycle
            # Change code accordingly
            for epoch in range(training_epochs):
                print("Epoch--->" + str(epoch))
                avg_cost = 0.
                total_batch = int(record_size / self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    print("batch--->" + str(i))
                    batch_x1, batch_ys = self.__createBatch(input1, labels, self.batch_size)


                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={self.x1: batch_x1, self.y: batch_ys})
                    print("cost per step-->"+str(c))
                    # Compute average loss
                    avg_cost += c / total_batch
                    count = count + self.batch_size
                # Display logs per epoch step
                if (epoch + 1) % display_step == 0:
                    # -1304 cost :0
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            #saver.save(sess, self.model_file)

        print("Optimization Finished!")

    def validateModel(self, test_input1, test_labels, one_hot_encoding=False):

        self.istraining = False
        test_labels = self.convertLabelsToOneHotVectors(test_labels)

        test_input1 = np.asarray(test_input1)
        test_labels = np.asarray(test_labels)

        print("Test1--->" + str(len(test_input1)))

        #test_input1, test_labels = self.reshape(test_input1, test_labels)



        print(len(test_input1))
        print(len(test_labels))
        record_size = len(test_input1)

        init_op = tf.global_variables_initializer()
        #saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            #saver.restore(sess, self.model_file)
            overall_accuracy = 0

            total_batch = int(record_size / self.batch_size)
            for i in range(total_batch):
                batch_x1, batch_ys = self.__createBatch(test_input1,  test_labels,
                                                                  self.batch_size)
                print(len(batch_x1))
                predictions = sess.run([self.pred], feed_dict={self.x1: batch_x1})
                # Compute Accuracy
                batch_log_loss = tf.losses.log_loss(predictions[0], batch_ys)
                print("Log Loss:", batch_log_loss.eval())


    def evaluateResults(self, predictions, actual):
        print(predictions)
        predictions = predictions[0]
        predicted = tf.equal(tf.argmax(predictions, 1), tf.argmax(actual, 1))
        batch_accuracy = tf.reduce_mean(tf.cast(predicted, "float"), name="accuracy")
        batch_accuracy = batch_accuracy.eval()

        return batch_accuracy

    def predict(self, test_input1):
        # Test model
        self.istraining = False
        result = []
        test_input1 = np.asarray(test_input1)
        record_size = len(test_input1)


        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            total_batch = int(record_size / self.batch_size) + 1

            for i in range(total_batch):
                batch_x1 = self.__createTestBatch(test_input1, batch_size=self.batch_size)

                print(len(batch_x1))
                predictions = sess.run([self.pred], feed_dict={self.x1: batch_x1})
                print(predictions)
                result.extend(predictions[0])
                # print(result)

        return result

    def generatePrediction(self,predictions):
        predicted = tf.argmax(predictions[0],1)
        predicted = predicted.eval()
        return predicted
