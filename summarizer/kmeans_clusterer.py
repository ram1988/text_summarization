from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import numpy as np

#https://stackoverflow.com/questions/26795535/output-50-samples-closest-to-each-cluster-center-using-scikit-learn-k-means-libr
#https://stackabuse.com/k-means-clustering-with-scikit-learn/
class TextSummarizer:
    def summarize(self,sentences,sentence_vectors,top_n):
        n_clusters = int(np.ceil(len(sentence_vectors) ** 0.5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans = kmeans.fit(sentence_vectors)
        closest = []
        '''
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        '''
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, \
                                                   sentence_vectors)
        print("summ....")
        print(len(sentence_vectors))
        print(n_clusters)
        print(closest)

        closest = sorted(closest)

        if top_n < len(closest):
            closest = closest[0:top_n]

        summary = ' '.join([sentences[idx] for idx in closest])

        print('Summarization Finished')
        return summary



