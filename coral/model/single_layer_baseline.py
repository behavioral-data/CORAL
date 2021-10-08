import torch.nn as nn
from .bert import BERT
import pdb
import torch

class Single(nn.Module):
    """
    Single layer baseline for CORAL that uses word2vec embeddings 
    """

    def __init__(self, vocab, word2vec, n_hidden, n_topics=5):
        super(Single, self).__init__()

        self.n_hidden = n_hidden
        self.n_topics = n_topics
        self.word2vec = word2vec

        self.dim_reduction = nn.Linear(self.n_hidden, self.n_topics)

        self.reconstruction = nn.Linear(self.n_topics, self.n_hidden, bias=False)

        self.spv_stage_label = nn.Linear(n_topics, 6)
        self.vocab = vocab


    def word2vec_graph(self,x):
        """Converts BERT inputs into matrix of Word2Vec embeddings"""
        rep = []   
        for i, key in enumerate(x):
            rep.append(self.word2vec.wv[self.vocab.idx2word[key]])
        return rep

    def forward(self, x, neg_x, segment_label, neg_segment_label, adj_mat, neg_adj_mat):

        token_vec = x
        import ipdb; ipdb.set_trace() 
        topic_dist = self.dim_reduction(token_vec)

        stage_vec = self.spv_stage_label(topic_dist)

        topic_dist = nn.Softmax(dim=1)(topic_dist)
        reconstructed_vec = self.reconstruction(topic_dist)

        #TODO: CHECK THIS
        neg_graph_vec = neg_x
        return reconstructed_vec, graph_vec, neg_graph_vec, topic_dist, stage_vec