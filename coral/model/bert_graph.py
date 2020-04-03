import torch.nn as nn

from .bert import BERT
import pdb


class BERTGraph(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size, markdown=False):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.markdown = markdown

    def forward(self, x, segment_label, adj_mat, train):
        x = self.bert(x, segment_label, adj_mat, train)
        # only return representation of '[CLS]' (graph representation)
        return x[:, 0]
