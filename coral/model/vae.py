from .bert_graph import BERTGraph
# from .markdown import BOW
import torch.nn as nn
from .bert import BERT
import pdb
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class TopicBERT(nn.Module):
    """calculate distribution over topics from code snippet representation"""

    def __init__(self, bert: BERT, vocab_size, n_topics=5):
        super(TopicBERT, self).__init__()
        # self.arg = arg
        self.n_topics = n_topics
        # self.bert = bert
        self.bert_graph = BERTGraph(bert, vocab_size)
        self.dim_reduction = nn.Linear(bert.hidden, self.n_topics)

    def forward(self, x, segment_label, adj_mat):
        graph_vec = self.bert_graph(x, segment_label, adj_mat)
        topic_dist = self.dim_reduction(graph_vec)
        topic_dist = nn.Softmax(dim=1)(topic_dist)
        # pdb.set_trace()
        return topic_dist, graph_vec
        # raise NotImplementedError


class VAE(nn.Module):
    """docstring for VAE"""

    def __init__(self, bert: BERT, vocab_size, markdown_vocab_size, markdown_emb_size, n_topics=5, weak_supervise=False, context=False, markdown=False):
        super(VAE, self).__init__()
        # self.arg = arg
        self.weak_supervise = weak_supervise
        self.context = context
        self.n_topics = n_topics
        self.markdown = markdown
        self.markdown_vocab_size = markdown_vocab_size
        self.markdown_emb_size = markdown_emb_size
        'TODO'
        '增加一个markdown 的embedding 模块'
        '把markdown 的embedding和graph的embedding concat 到一起'
        if markdown:
            self.markdown_emb = BOW(markdown_vocab_size, markdown_emb_size, 2)
        # self.bert = bert
        self.bert_graph = BERTGraph(bert, vocab_size, markdown)
        if markdown:
            self.dim_reduction = nn.Linear(
                bert.hidden + markdown_emb_size, self.n_topics)
        else:
            self.dim_reduction = nn.Linear(bert.hidden, self.n_topics)

        # self.topic_bert = TopicBERT(bert, vocab_size, n_topics)
        if markdown:
            self.reconstruction = nn.Linear(
                n_topics, bert.hidden + markdown_emb_size, bias=False)
        else:
            self.reconstruction = nn.Linear(n_topics, bert.hidden, bias=False)
        # raise NotImplementedError
        if weak_supervise:
            self.spv_stage_label = nn.Linear(n_topics, 6)
        if context:
            self.context_stage_label = nn.Linear(n_topics, 6).double()
        # pdb.set_trace()

    def forward(self, x, neg_x, segment_label, neg_segment_label, adj_mat, neg_adj_mat, train, context_topic_dist, markdown_label, markdown_len, neg_markdown_label, neg_markdown_len):
        # topic_dist, graph_vec = self.topic_bert(x, segment_label, adj_mat)

        graph_vec = self.bert_graph(x, segment_label, adj_mat, train)
        # pdb.set_trace()
        if self.markdown:
            # pdb.set_trace()
            markdown_vec = self.markdown_emb(markdown_label, markdown_len)
            graph_vec = torch.cat((graph_vec, markdown_vec), dim=1)
            # raise NotImplementedError
        topic_dist = self.dim_reduction(graph_vec)
        if self.weak_supervise:
            stage_vec = self.spv_stage_label(topic_dist)
            if self.context:
                context_topic_dist = torch.sum(context_topic_dist, dim=1)
                # pdb.set_trace()
                context_stage_vec = self.context_stage_label(
                    context_topic_dist)
                stage_vec += context_stage_vec
        else:
            stage_vec = None
        topic_dist = nn.Softmax(dim=1)(topic_dist)

        reconstructed_vec = self.reconstruction(topic_dist)
        neg_graph_vec = self.bert_graph(
            neg_x, neg_segment_label, neg_adj_mat, train)
        if self.markdown:
            neg_markdown_vec = self.markdown_emb(
                neg_markdown_label, neg_markdown_len)
            neg_graph_vec = torch.cat((neg_graph_vec, neg_markdown_vec), dim=1)
        # pdb.set_trace()
        return reconstructed_vec, graph_vec, neg_graph_vec, topic_dist, stage_vec


class TempVAE(nn.Module):
    """
    append annotation after code
    one transformer
    """

    def __init__(self, bert: BERT, vocab_size, n_topics=5, weak_supervise=False, context=False, markdown=False):
        super(TempVAE, self).__init__()
        # self.arg = arg

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
        # raise NotImplementedError

        self.spv_stage_label = nn.Linear(n_topics, 6)

    def forward(self, x, neg_x, segment_label, neg_segment_label, adj_mat, neg_adj_mat, train):

        graph_vec = self.bert_graph(x, segment_label, adj_mat, train)
        topic_dist = self.dim_reduction(graph_vec)

        stage_vec = self.spv_stage_label(topic_dist)
        # return None, None, None, None, stage_vec
        topic_dist = nn.Softmax(dim=1)(topic_dist)
        reconstructed_vec = self.reconstruction(topic_dist)

        neg_graph_vec = self.bert_graph(
            neg_x, neg_segment_label, neg_adj_mat, train)
        return reconstructed_vec, graph_vec, neg_graph_vec, topic_dist, stage_vec
