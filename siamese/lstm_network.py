import tensorflow as tf
from tensorflow.keras import layers



#https://towardsdatascience.com/first-contact-with-tensorflow-estimator-69a5e072998d
	#https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/RNNEstimator
	class SentenceEncoder:

		def __init__(self,vector_size, auth_classes, embed_matrix):
			self.vector_size = vector_size
			self.auth_classes = auth_classes
			self.embedding_matrix = embed_matrix


		def buildRNN(self, x, scope):
			print(x)
			#bidirectional rnn to get the sentence vectors
			#https://heartbeat.fritz.ai/using-a-keras-embedding-layer-to-handle-text-data-2c88dc019600
			model = Sequential()
			model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5,
																				  10)))
			model.add(Dense(5))
			model.add(Activation('softmax'))



		def __model_fn(self,features, labels, mode, params):

			self.logits = self.buildRNN(features)
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


		def __train_model_fn(self,image_labels,mode,params,logits,loss,train_op):
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