from .bert_graph import BERTGraph

import torch.nn as nn
from .bert import BERT
import pdb
import torch


class CORAL(nn.Module):
    """
    append annotation after code
    one transformer
    """

    def __init__(self, bert: BERT, vocab_size, n_topics=5, weak_supervise=False, context=False, markdown=False):
        super(CORAL, self).__init__()

        self.weak_supervise = weak_supervise
        self.context = context
        self.n_topics = n_topics
        self.markdown = markdown

        'TODO'
        '增加一个markdown 的embedding 模块'
        '把markdown 的embedding和graph的embedding concat 到一起'

        self.bert_graph = BERTGraph(bert, vocab_size, markdown)

        self.dim_reduction = nn.Linear(bert.hidden, self.n_topics)

        self.reconstruction = nn.Linear(n_topics, bert.hidden, bias=False)

        self.spv_stage_label = nn.Linear(n_topics, 6)

    def forward(self, x, neg_x, segment_label, neg_segment_label, adj_mat, neg_adj_mat, train):

        graph_vec = self.bert_graph(x, segment_label, adj_mat, train)
        topic_dist = self.dim_reduction(graph_vec)

        stage_vec = self.spv_stage_label(topic_dist)

        topic_dist = nn.Softmax(dim=1)(topic_dist)
        reconstructed_vec = self.reconstruction(topic_dist)

        neg_graph_vec = self.bert_graph(
            neg_x, neg_segment_label, neg_adj_mat, train)
        return reconstructed_vec, graph_vec, neg_graph_vec, topic_dist, stage_vec
