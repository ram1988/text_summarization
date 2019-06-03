# Text Summarization

Trained the RNN LSTM model to generate the Sentence Encoder(SE).

Sentences would be passed to the SE to generate the sentence vectors. 

With the sentence vectors, k-means clustering is performed to infer the top 'n' sentences which is closer to the cluster centroids. 

Hence, the text summary is generated. 
