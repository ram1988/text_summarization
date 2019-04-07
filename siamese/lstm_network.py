import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


#https://towardsdatascience.com/first-contact-with-tensorflow-estimator-69a5e072998d
#https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/RNNEstimator
class SentenceEncoder:

	def __init__(self,vector_size, img_classes, embed_matrix):
		self.vector_size = vector_size
		self.img_classes = img_classes
        self.embedding_matrix = embed_matrix


    def buildRNN(self, x, scope):
        print(x)
		#no of layers based on the no. of sentences
        x = tf.transpose(x, [1, 0, 2])

        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            fw_cell_array = []
            print(tf.get_variable_scope().name)
            for _ in range(self.n_layers):
                fw_cell = rnn.GRUCell(self.n_hidden, activation=tf.nn.relu)
                fw_cell_array.append(fw_cell)
            fw_cell = rnn.MultiRNNCell(fw_cell_array, state_is_tuple=True)
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            bw_cell_array = []
            print(tf.get_variable_scope().name)
            for _ in range(self.n_layers):
                bw_cell = rnn.GRUCell(self.n_hidden, activation=tf.nn.relu)
                bw_cell_array.append(bw_cell)
            bw_cell = rnn.MultiRNNCell(bw_cell_array, state_is_tuple=True)

        outputs = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float64, time_major=True)
        print("output-->" + str(outputs))
        outputs = tf.concat(outputs[0], 2)
        outputs = tf.reshape(outputs, [-1, self.nfeatures])
        outputs = tf.split(outputs, self.n_steps, 0)
        print("output-->" + str(outputs))
        outputs = outputs[-1]
        print("output-->" + str(outputs))

        nn_layer1 = tf.layers.dense(outputs, 1024, activation=tf.nn.relu)
        nn_layer1 = tf.layers.dropout(nn_layer1, rate=0.8)
        nn_layer2 = tf.layers.dense(nn_layer1, 1024, activation=tf.nn.relu)
        nn_layer2 = tf.layers.dropout(nn_layer2, rate=0.8)
        result = tf.layers.dense(nn_layer2, 3, activation=tf.nn.softmax)

        print("final result11-->" + str(result))

        return result



	def __model_fn(self,features, labels, mode, params):

		self.logits = self.buildRNN(img_features)
		loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=self.logits)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_or_create_global_step())


		if mode == tf.estimator.ModeKeys.TRAIN:
			return self.__train_model_fn(labels, mode, params, self.logits, loss, train_op)
		elif mode == tf.estimator.ModeKeys.EVAL:
			print("evaluate...111")
			obj = self.__eval_model_fn(labels,self.logits,loss)
			print("val ends")
			return obj
		else:
			return self.__predict_model_fn(logits)

	def __train_model_fn(self,image_labels,mode,params,logits,loss,train_op):
		print(mode)
		print("training....")
		print(image_labels)
		image_labels = tf.cast(image_labels, tf.float32)
		print(image_labels.shape)
		print(tf.size(image_labels))

		#loss = tf.losses.softmax_cross_entropy(onehot_labels=image_labels, logits=logits)
		# Configure the Training Op (for TRAIN mode)

		return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)


	def __eval_model_fn(self,image_labels,logits,loss):
		image_labels = tf.cast(image_labels, tf.float32)
		print("eval model...")
		print(logits)
		#loss = tf.losses.softmax_cross_entropy(onehot_labels=image_labels, logits=logits)
		# Add evaluation metrics (for EVAL mode)
		predictions = {
			# Generate predictions (for PREDICT and EVAL mode)
			"classes": tf.argmax(input=logits, axis=1),
			# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
			# `logging_hook`.
			"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
		}
		print("predictions....")
		eval_metric_ops = {
				"accuracy": tf.metrics.accuracy(
					labels=tf.argmax(input=image_labels, axis=1), predictions=predictions["classes"])}
		print(image_labels)
		return tf.estimator.EstimatorSpec(
				mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_ops)

	def __predict_model_fn(self,logits):
		print("PRED....")
		print(logits)
		predictions = {
				"classes": tf.argmax(logits, axis=1),
				"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
			}
		return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
										  predictions=predictions,
										  export_outputs={
											  'classify': tf.estimator.export.PredictOutput(predictions)
										  })


	def get_classifier_model(self):
		print("get the model...")
		return tf.estimator.Estimator(
			model_fn = self.__model_fn, model_dir="/tmp/cnn_data")