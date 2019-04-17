# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import nltk

	class SentenceEncoder:

		def __init__(self):
			#dd

		def preprocess(self,data):
			sentences = []
			for i in data['body'].values:
				for j in nltk.sent_tokenize(i):
					sentences.append(j)

			#preprocess for keras
			num_words=50000
			maxlen=50
			tokenizer = Tokenizer(num_words = num_words, split=' ')
			tokenizer.fit_on_texts(sentences)
			seqs = tokenizer.texts_to_sequences(sentenses)
			pad_seqs = []
			for i in seqs:
				if len(i)>4:
					pad_seqs.append(i)
			pad_seqs = pad_sequences(pad_seqs,maxlen)


		def trainModel(self, x, scope):
			#The model
			embed_dim = 150
			latent_dim = 128
			batch_size = 64

			#### Encoder Model ####
			encoder_inputs = Input(shape=(maxlen,), name='Encoder-Input')
			emb_layer = Embedding(num_words, embed_dim,input_length = maxlen, name='Body-Word-Embedding', mask_zero=False)
			# Word embeding for encoder (ex: Issue Body)
			x = emb_layer(encoder_inputs)
			bi_directional_encoder = Sequential()
    		bi_directional_encoder.add(Bidirectional(LSTM(10, return_sequences=True, name='Encoder-Last-LSTM'), input_shape=(5,10)))

			encoder_model = Model(inputs=encoder_inputs, outputs=bi_directional_encoder, name='Encoder-Model')
			seq2seq_encoder_out = encoder_model(encoder_inputs)
			#### Decoder Model ####
			decoded = RepeatVector(maxlen)(seq2seq_encoder_out)
			decoder_gru = GRU(latent_dim, return_sequences=True, name='Decoder-GRU-before')
			decoder_gru_output = decoder_gru(decoded)
			decoder_dense = Dense(num_words, activation='softmax', name='Final-Output-Dense-before')
			decoder_outputs = decoder_dense(decoder_gru_output)
			#### Seq2Seq Model ####
			#seq2seq_decoder_out = decoder_model([decoder_inputs, seq2seq_encoder_out])
			seq2seq_Model = Model(encoder_inputs,decoder_outputs )
			seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')
			history = seq2seq_Model.fit(pad_seqs, np.expand_dims(pad_seqs, -1),
					batch_size=batch_size,
					epochs=5,
					validation_split=0.12)

