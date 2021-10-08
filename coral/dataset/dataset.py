import json
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import random

import numpy as np
import itertools

import re

from .vocab import UnitedVocab


random.seed(1111)

STAGE_PAD = 0
WRANGLE = 1
EXPLORE = 2
MODEL = 3
EVALUATE = 4
IMPORT = 5

SPV_MODE = [STAGE_PAD, WRANGLE, EXPLORE, MODEL, EVALUATE, IMPORT]


wrangle_funcs = ['pandas.read_csv', 'pandas.read_csv.dropna', 'pandas.read_csv.fillna',
                 'pandas.DataFrame.fillna', 'sklearn.datasets.load_iris', 'scipy.misc.imread',
                 'scipy.io.loadmat', 'sklearn.preprocessing.LabelEncoder', 'scipy.interpolate.interp1d']

explore_funcs = ['seaborn.distplot', 'matplotlib.pyplot.show', 'matplotlib.pyplot.plot', 'matplotlib.pyplot.figure',
                 'seaborn.pairplot', 'seaborn.heatmap', 'seaborn.lmplot', 'pandas.read_csv.describe',
                 'pandas.DataFrame.describe']

model_funcs = ['sklearn.cluster.KMeans',
               'sklearn.decomposition.PCA',
               'sklearn.naive_bayes.GaussianNB',
               'sklearn.ensemble.RandomForestClassifier',
               'sklearn.linear_model.LinearRegression',
               'sklearn.linear_model.LogisticRegression',
               'sklearn.tree.DecisionTreeRegressor',
               'sklearn.ensemble.BaggingRegressor',
               'sklearn.neighbors.KNeighborsClassifier',
               'sklearn.naive_bayes.MultinomialNB',
               'sklearn.svm.SVC',
               'sklearn.tree.DecisionTreeClassifier',
               'tensorflow.Session',
               'sklearn.linear_model.Ridge',
               'sklearn.linear_model.Lasso']

evaluate_funcs = ['sklearn.metrics.confusion_matrix', 'sklearn.cross_validation.cross_val_score',
                  'sklearn.metrics.mean_squared_error', 'sklearn.model_selection.cross_val_score', 'scipy.stats.ttest_ind', 'sklearn.metrics.accuracy_score']


def split_func_name(func):
    """
    split function names
    eg. sklearn.metrics.pairwise.cosine_similarity -> [sklearn, metrics, pairwise, cosine, similarity]
    """
    if ' ' in func:
        return func.split()
    new_str = ''
    for i, l in enumerate(func):
        if i > 0 and l.isupper() and func[i - 1].islower():
            new_str += '.'
        elif i > 0 and i < len(func) - 1 and l.isupper() and func[i - 1].isupper() and func[i + 1].islower():
            new_str += '.'
        elif i > 0 and l.isdigit() and func[i - 1].isalpha():
            new_str += '.'
        elif i < len(func) - 1 and l.isalpha() and func[i - 1].isdigit():
            new_str += '.'
        else:
            pass
        new_str += l
    return re.split('\.|_|/', new_str.lower())


def cell_type(funcs, nodes=None, header=None):
    """
    5 heuristic rules for weak supervision
    """
    grams = [t.lower() for t in header.split() if t]
    bi_grams = ['{} {}'.format(t, grams[i + 1])
                for i, t in enumerate(grams[:-1])]

    if sum([1 for n in nodes if (n["type"] == 'Import' or n["type"] == 'ImportFrom')]) / len(nodes) > 0.3:
        return IMPORT

    if any([g in bi_grams for g in ['logistic regression', 'machine learning', 'random forest']]) and len(grams) <= 3:
        return MODEL
    if 'cross validation' in bi_grams and len(grams) <= 3:
        return EVALUATE

    if any([f in funcs for f in model_funcs]):
        return MODEL
    if any([f in funcs for f in evaluate_funcs]):
        return EVALUATE
    if any([f in funcs for f in explore_funcs]):
        return EXPLORE
    if len(nodes) == 3 and nodes[1]["type"] == "Expr":
        return EXPLORE

    if any([f in funcs for f in wrangle_funcs]):
        return WRANGLE

    return STAGE_PAD

def multi_cell_type(funcs, nodes = None, header = None):
    
    grams = [t.lower() for t in header.split() if t]
    bi_grams = ['{} {}'.format(t, grams[i + 1])
                for i, t in enumerate(grams[:-1])]
    
    labels = []

    if sum([1 for n in nodes if (n["type"] == 'Import' or n["type"] == 'ImportFrom')]) / len(nodes) > 0.3:
        labels.append(IMPORT)

    if any([g in bi_grams for g in ['logistic regression', 'machine learning', 'random forest']]) and len(grams) <= 3:
        labels.append(MODEL)
    elif any([f in funcs for f in model_funcs]):
        labels.append(MODEL)


    if 'cross validation' in bi_grams and len(grams) <= 3:
        labels.append(EVALUATE)
    elif any([f in funcs for f in evaluate_funcs]):
        labels.append(EVALUATE)
    
    if any([f in funcs for f in explore_funcs]):
        labels.append(EXPLORE)
    elif len(nodes) == 3 and nodes[1]["type"] == "Expr":
        labels.append(EXPLORE)

    if any([f in funcs for f in wrangle_funcs]):
        labels.append(WRANGLE)

    return labels

class DataReader(object):
    """
    read graphs from file
    split tokens into subtokens and create a more fine-grained adjcency matrix
    """

    def __init__(self, graph_path, shuffle=False, seq_len=None, max_graph_num=None):
        super(DataReader, self).__init__()

        self.graph_path = graph_path

        self.seq_len = seq_len

        self.max_graph_num = max_graph_num

        graphs = []
        with open(graph_path, 'r', encoding='utf-8') as f:
            for l in tqdm(f):
                g = json.loads(l)

                new_nodes = []
                idx_map = {}
                for i, n in enumerate(g["nodes"]):
                    if "value" not in n:
                        tokens = [n["type"]]
                    else:

                        tokens = split_func_name(n["value"])
                        tokens = [t for t in tokens if t]

                    idx_map[i] = []
                    for t in tokens:
                        idx_map[i].append(len(new_nodes))
                        new_nodes.append(t)
                g["new_nodes"] = new_nodes
                g["idx_map"] = idx_map
                graphs.append(g)

                if max_graph_num and len(graphs) > max_graph_num:
                    break
        if seq_len:
            graphs = [g for g in graphs if len(
                g["nodes"]) + 1 <= seq_len]

        if shuffle:
            random.shuffle(graphs)
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)


class CORALDataset(Dataset):

    def __init__(self, graphs, vocab, seq_len, chunk_size=128, n_neg=5, markdown=False, weak_labeler = cell_type):
        self.vocab = vocab
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.n_neg = n_neg
        self.markdown = markdown
        self.weak_labeler = weak_labeler
        self.word2vec = None

        if markdown:
            graphs = list(sorted(graphs, key=lambda x: len(
                x["new_nodes"]) + len([t for t in ' '.join(x["annotation"]).split() if t])))
        else:
            graphs = list(
                sorted(graphs, key=lambda x: len(x["new_nodes"])))

        graphs = [graphs[i:i + chunk_size]
                  for i in range(0, len(graphs), chunk_size)]
        random.shuffle(graphs)
        graphs = list(itertools.chain.from_iterable(graphs))
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        def process_graph(graph):

            # get sequence
            t1_label = self.get_code_seq(graph)

            if self.markdown:
                t2_label = self.get_markdown_seq(graph)
                bert_input = [self.vocab.sos_index] + t1_label + \
                    [self.vocab.sep_index] + t2_label + [self.vocab.eos_index]
                segment_label = [1] * (2 + len(t1_label)) + \
                    [2] * (len(t2_label) + 1)
            else:
                bert_input = [self.vocab.sos_index] + \
                    t1_label + [self.vocab.eos_index]
                segment_label = [1] * (2 + len(t1_label))
            assert len(bert_input) == len(segment_label)
            bert_input = bert_input[:self.seq_len]
            segment_label = segment_label[:self.seq_len]

            # get adjcency matrix
            if self.markdown:
                adj_mat = np.zeros(
                    (3 + len(t1_label) + len(t2_label), 3 + len(t1_label) + len(t2_label)))
            else:
                adj_mat = np.zeros(
                    (len(graph["new_nodes"]) + 2, len(graph["new_nodes"]) + 2))

            for i, n in enumerate(graph["nodes"]):
                if "children" not in n:
                    continue
                for c in n["children"]:

                    for ii in [z + 1 for z in graph["idx_map"][i]]:
                        for cc in [z + 1 for z in graph["idx_map"][c]]:
                            adj_mat[ii][cc] = 1
                            adj_mat[cc][ii] = 1

            adj_mat[:, 0] = 1
            adj_mat[0, :] = 1
            if self.markdown:
                adj_mat[2 + len(t1_label):, :] = 1
                adj_mat[:, 2 + len(t1_label):] = 1
            adj_mat = adj_mat[:self.seq_len, :self.seq_len]
            if graph["funcs"] is not None:
                stage = self.weak_labeler(
                    graph["funcs"], nodes=graph["nodes"], header=graph["header"])

            else:
                stage = int(graph["stage"])

           
            output = {"bert_input": bert_input,
                      "segment_label": segment_label,
                      "adj_mat": adj_mat,
                      "stage": stage,
                      "seq_len": len(bert_input)}
            
            if self.word2vec:
                print("Has word2vec!")
                rep = []   
                for i, key in enumerate(bert_input):
                    rep.append(self.word2vec.wv[self.vocab.idx2word[key]])                
                output["word2vec_embedding"] = rep

            return output

        pos_output = process_graph(self.graphs[item])
        neg_ids = random.sample(range(len(self.graphs)), self.n_neg)

        neg_output = [process_graph(
            self.graphs[i]) for i in neg_ids]

        return pos_output, neg_output

    def get_code_seq(self, graph):

        nodes = graph["new_nodes"]
        ids = [self.vocab.word2idx.get(n, self.vocab.unk_index) for n in nodes]
        return ids

    def get_markdown_seq(self, graph):
        markdown = graph["annotation"]
        markdown = " [SEP] ".join([l.lower() for l in markdown if l.strip()])

        tokens = [t for t in markdown.split() if t]
        ids = [self.vocab.word2idx.get(
            t, self.vocab.unk_index) for t in tokens]
        return ids


def my_collate(batch):
    def collate_graphs(graphs):

        seq_len = max([item["seq_len"] for item in graphs])

        bert_input = []

        segment_label = []
        adj_mat = []
        stages = []
        word2vec_embeddings = []

        for item in graphs:
            item["bert_input"] += [UnitedVocab.pad_index] * \
                (seq_len - item["seq_len"])

            item["segment_label"] += [UnitedVocab.pad_index] * \
                (seq_len - item["seq_len"])
            mat = np.zeros((seq_len, seq_len))
            mat[:item["adj_mat"].shape[0],
                :item["adj_mat"].shape[1]] = item["adj_mat"]
            
            if "word2vec_embedding" in item:
                item["word2vec_embedding"] += [UnitedVocab.pad_index] * \
                (seq_len - item["seq_len"])
                word2vec_embeddings.apppend(item["word2vec_embedding"])

            bert_input.append(item["bert_input"])
            segment_label.append(item["segment_label"])
            adj_mat.append(mat)
            stages.append(item["stage"])

        to_return =  {"bert_input": torch.tensor(bert_input),
                "segment_label": torch.tensor(segment_label),
                "adj_mat": torch.tensor(adj_mat),
                "stage": torch.tensor(stages)}
        if len(word2vec_embeddings) > 0:
            to_return["word2vec_embeddings"] = torch.tensor(word2vec_embeddings)
        
        return to_return
    pos_graphs = [item[0] for item in batch]
    neg_graphs = list(itertools.chain.from_iterable(
        [item[1] for item in batch]))

    return collate_graphs(pos_graphs), collate_graphs(neg_graphs)
