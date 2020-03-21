import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import random
import pdb
import numpy as np
import itertools
import os
import re
from collections import OrderedDict


# from torch.utils.data import

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
# 'matplotlib.pyplot.xlabel', 'matplotlib.pyplot.ylabel'
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
  # pdb.set_trace()
  # print(header)
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
    # print(h)
  return STAGE_PAD




class DataReader(object):
    """docstring for DataReader"""

    def __init__(self, graph_path, graphs=None, shuffle=False, duplicate=1, seq_len=None, use_sub_token=False, max_graph_num = None):
        super(DataReader, self).__init__()
        # self.arg = arg
        self.graph_path = graph_path
        self.duplicate = duplicate
        self.seq_len = seq_len
        self.use_sub_token = use_sub_token
        self.max_graph_num=max_graph_num
        if graphs is None:

            graphs = []
            with open(graph_path, 'r', encoding='utf-8') as f:
                for l in tqdm(f):
                    g = json.loads(l)
                    # if "annotation" not in g:
                    #     continue
                    if use_sub_token:
                        new_nodes = []
                        idx_map = {}
                        for i, n in enumerate(g["nodes"]):
                            if "value" not in n:
                                tokens = [n["type"]]
                            else:

                                tokens = split_func_name(n["value"])
                                tokens = [t for t in tokens if t]
                                # if len(tokens) > 1:
                                #     print(n["value"], tokens)
                            idx_map[i] = []
                            for t in tokens:
                                idx_map[i].append(len(new_nodes))
                                new_nodes.append(t)
                        g["new_nodes"] = new_nodes
                        g["idx_map"] = idx_map
                        graphs.append(g)
                        # raise NotImplementedError
                    else:
                        graphs.append(g)
                    if max_graph_num and len(graphs) > max_graph_num:
                        break
        if seq_len:
            graphs = [g for g in graphs if len(
                g["nodes"]) + 1 <= seq_len]

        graphs = graphs * duplicate  # which actually is not excecuted

        if shuffle:
            random.shuffle(graphs)
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)


class Vocab(object):
    """docstring for Vocab"""
    mask_index = 0
    unk_index = 1
    pad_index = 2
    sos_index = 3
    eos_index = 4

    def __init__(self, graphs, use_sub_token=False, min_occur=3, path=None):
        super(Vocab, self).__init__()
        self.mask_index = 0
        self.unk_index = 1
        self.pad_index = 2
        self.sos_index = 3
        self.eos_index = 4
        self.use_sub_token = use_sub_token
        self.min_occur = min_occur

        counter = {}
        for g in tqdm(graphs):
            if use_sub_token:
                for n in g["new_nodes"]:
                    if n not in counter:
                        counter[n] = 0
                    counter[n] += 1
            else:
                for n in g["nodes"]:
                    token = n["type"] if "value" not in n else n["value"]
                    if token not in counter:
                        counter[token] = 0
                    counter[token] += 1
        idx2word = ["[MASK]", "[UNK]", "[PAD]", "[CLS]", "[SEP]"] + \
            [w for w in counter if counter[w] >= min_occur]
        word2idx = {w: i for i, w in enumerate(idx2word)}
        with open('./vocab.txt','r') as f:
            data = json.load(f)
            idx2word = data["idx2word"]
            word2idx  = data["word2idx"]
            # json.dump({"idx2word":idx2word,
            #     "word2idx":word2idx}, fout)
        self.idx2word = idx2word
        self.word2idx = word2idx

    def __len__(self):
        return len(self.idx2word)


class CodeVocab(object):
    """docstring for Vocab"""
    mask_index = 0
    unk_index = 1
    pad_index = 2
    sos_index = 3
    eos_index = 4

    def __init__(self, graphs, use_sub_token=False, min_occur=3, path=None, max_size=10000):
        super(CodeVocab, self).__init__()
        self.mask_index = 0
        self.unk_index = 1
        self.pad_index = 2
        self.sos_index = 3
        self.eos_index = 4
        self.use_sub_token = use_sub_token
        self.min_occur = min_occur
        self.max_size = max_size

        counter = {}
        for g in tqdm(graphs):
            if use_sub_token:
                for n in g["new_nodes"]:
                    if n not in counter:
                        counter[n] = 0
                    counter[n] += 1
            else:
                for n in g["nodes"]:
                    token = n["type"] if "value" not in n else n["value"]
                    if token not in counter:
                        counter[token] = 0
                    counter[token] += 1
        selected_tokens = [w for w in counter if counter[w] >= min_occur]
        selected_tokens = list(
            sorted(selected_tokens, key=lambda x: -counter[x]))
        # pdb.set_trace()
        idx2word =     selected_tokens
        idx2word = idx2word[:max_size]
        word2idx = {w: i for i, w in enumerate(idx2word)}

        self.idx2word = idx2word
        self.word2idx = word2idx

    def __len__(self):
        return len(self.idx2word)


class MarkdownVocab(object):
    """docstring for Vocab"""
    mask_index = 0
    unk_index = 1
    pad_index = 2
    sos_index = 3
    eos_index = 4

    def __init__(self, graphs, use_sub_token=False, min_occur=3, path=None):
        super(MarkdownVocab, self).__init__()
        self.mask_index = 0
        self.unk_index = 1
        self.pad_index = 2
        self.sos_index = 3
        self.eos_index = 4
        self.ept_index = 5
        self.use_sub_token = use_sub_token
        self.min_occur = min_occur

        counter = {}
        for g in tqdm(graphs):
            header = g["header"]
            token = [t.lower() for t in header.split() if t]
            for t in token:
                if t not in counter:
                    counter[t] = 0
                counter[t] += 1

        idx2word = ["[MASK]", "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[EPT]"] + \
            [w for w in counter if counter[w] >= min_occur]
        word2idx = {w: i for i, w in enumerate(idx2word)}
        with open('./md_vocab.txt','r') as f:
            data = json.load(f)
            idx2word = data["idx2word"]
            word2idx = data["word2idx"]
            # json.dump({"idx2word":idx2word,
            #     "word2idx":word2idx}, fout)

        self.idx2word = idx2word
        self.word2idx = word2idx

    def __len__(self):
        return len(self.idx2word)


class AnnotationVocab(object):
    """
    vocab for annotation
    """
    mask_index = 0
    unk_index = 1
    pad_index = 2
    sos_index = 3
    eos_index = 4

    def __init__(self, graphs, use_sub_token=False, min_occur=3, path=None, max_size=10000):
        super(AnnotationVocab, self).__init__()
        self.mask_index = 0
        self.unk_index = 1
        self.pad_index = 2
        self.sos_index = 3
        self.eos_index = 4
        self.ept_index = 5
        self.use_sub_token = use_sub_token
        self.min_occur = min_occur
        self.max_size = max_size

        counter = {}
        for g in tqdm(graphs):
            # header = g["annotation"]
            header = g["annotation"][-1] if len(g["annotation"]) > 0 else ""
            token = [t.lower() for t in header.split() if t]
            for t in token:
                if t not in counter:
                    counter[t] = 0
                counter[t] += 1
        selected_tokens = [w for w in counter if counter[w] >= min_occur]
        selected_tokens = list(
            sorted(selected_tokens, key=lambda x: -counter[x]))

        idx2word =             selected_tokens
        idx2word = idx2word[:max_size]

        word2idx = {w: i for i, w in enumerate(idx2word)}

        self.idx2word = idx2word
        self.word2idx = word2idx

    def __len__(self):
        return len(self.idx2word)


class UnitedVocab(object):
    """
    vocab for annotation
    """
    pad_index  = 0
    unk_index = 1
    mask_index = 2
    sos_index = 3
    sep_index = 4
    eos_index = 5

    def __init__(self, graphs, use_sub_token=False, min_occur=3, path=None):
        super(UnitedVocab, self).__init__()
        self.pad_index = 0
        self.unk_index = 1
        self.mask_index = 2
        self.sos_index = 3
        self.sep_index = 4
        self.eos_index = 5

        self.use_sub_token = use_sub_token
        self.min_occur = min_occur

        if not os.path.exists(path):
            code_vocab = CodeVocab(
                graphs, use_sub_token=use_sub_token, min_occur=min_occur)
            annotation_vocab = AnnotationVocab(
                graphs, use_sub_token=use_sub_token, min_occur=min_occur)
            idx2word =list(set(code_vocab.idx2word+annotation_vocab.idx2word)) # 这一步每一次的生成结果不一样，需要在后面排序固定字典

            # pdb.set_trace()


            idx2word = ["[PAD]", "[UNK]","[MASK]" , "[CLS]", "[SEP]", "[EOS]"] + \
                idx2word
            word2idx = {w: i for i, w in enumerate(idx2word)}
            with open(path, 'w') as fout:
                json.dump({"idx2word":idx2word,
                    "word2idx":word2idx}, fout)

        else:
            with open(path,'r') as f:
                dictionary = json.load(f)
                idx2word = dictionary["idx2word"]
                word2idx = dictionary["word2idx"]
            # raise NotImplementedError

        self.idx2word = idx2word
        self.word2idx = word2idx

    def __len__(self):
        return len(self.idx2word)



class SNAPVocab(object):
    """docstring for Vocab"""
    pad_index  = 0
    unk_index = 1
    mask_index = 2
    sos_index = 3
    sep_index = 4
    eos_index = 5

    def __init__(self, graphs, use_sub_token=False, min_occur=3, path=None, max_size=10000):
        super(SNAPVocab, self).__init__()
        self.pad_index = 0
        self.unk_index = 1
        self.mask_index = 2
        self.sos_index = 3
        self.sep_index = 4
        self.eos_index = 5
        self.use_sub_token = use_sub_token
        self.min_occur = min_occur
        self.max_size = max_size

        counter = {}
        for g in tqdm(graphs):
            if use_sub_token:
                for n in g["new_nodes"]:
                    if n not in counter:
                        counter[n] = 0
                    counter[n] += 1
            else:
                for n in g["nodes"]:
                    token = n["type"] if "value" not in n else n["value"]
                    if token not in counter:
                        counter[token] = 0
                    counter[token] += 1
        selected_tokens = [w for w in counter if counter[w] >= min_occur]
        selected_tokens = list(
            sorted(selected_tokens, key=lambda x: -counter[x]))
        # pdb.set_trace()
        idx2word =    ["[PAD]", "[UNK]","[MASK]" , "[CLS]", "[SEP]", "[EOS]"] +  selected_tokens
        idx2word = idx2word[:max_size]
        word2idx = {w: i for i, w in enumerate(idx2word)}

        self.idx2word = idx2word
        self.word2idx = word2idx

    def __len__(self):
        return len(self.idx2word)



class CustomBERTDataset(Dataset):
    def __init__(self, graphs, vocab, markdown_vocab, seq_len, use_sub_token=False, encoding="utf-8", corpus_lines=None, on_memory=True, chunk_size=128, n_neg=5, n_topics=50):
        self.vocab = vocab
        self.markdown_vocab = markdown_vocab
        self.seq_len = seq_len if not use_sub_token else int(seq_len * 1.5)

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines

        self.encoding = encoding
        self.chunk_size = chunk_size
        self.use_sub_token = use_sub_token
        self.n_neg = n_neg
        self.n_topics = n_topics

        self.neighbor_topic_dist = {
            g["id"]: np.zeros(n_topics) for g in graphs}
        # graphs = [g for g in graphs if len(g["nodes"]) + 1 <= seq_len]
        if use_sub_token:
            graphs = list(sorted(graphs, key=lambda x: len(x["new_nodes"])))
            # raise NotImplementedError
        else:
            graphs = list(sorted(graphs, key=lambda x: len(x["nodes"])))
        graphs = [graphs[i:i + chunk_size]
                  for i in range(0, len(graphs), chunk_size)]
        random.shuffle(graphs)
        graphs = list(itertools.chain.from_iterable(graphs))
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)
        # return self.corpus_lines

    def __getitem__(self, item):
        def process_graph(graph):
            # output 添加一个 neighbor topic 的属性
            t1_random, t1_label = self.random_word(graph)
            t1 = [self.vocab.sos_index] + t1_random  # + [self.vocab.eos_index]
            t1_label = [self.vocab.pad_index] + \
                t1_label  # + [self.vocab.pad_index]
            segment_label = [1 for _ in range(len(t1))][:self.seq_len]
            bert_input = t1[:self.seq_len]
            bert_label = t1_label[:self.seq_len]
            # assert len(bert_input) == len(graph["new_nodes"]) + 1
            adj_mat = np.zeros((len(bert_input), len(bert_input)))
            for i, n in enumerate(graph["nodes"]):
                if "children" not in n:
                    continue
                for c in n["children"]:
                    if self.use_sub_token:
                        pass
                        # for ii in graph["idx_map"][i]:
                        #     for cc in graph["idx_map"][c]:
                        #         if ii + 1 >= len(bert_input) or cc + 1 >= len(bert_input):
                        #             continue
                        #         # assert cc < len(graph["new_nodes"])
                        #         adj_mat[ii + 1][cc + 1] = 1
                        # adj_mat[[z + 1 for z in graph["idx_map"][i]],
                        #         [z + 1 for z in graph["idx_map"][c]]] = 1
                        # raise NotImplementedError
                    else:
                        adj_mat[i + 1][c + 1] = 1
            adj_mat[:len(
                graph["nodes"]) + 1 if not self.use_sub_token else len(graph["new_nodes"]) + 1, 0] = 1
            adj_mat[0, :len(
                graph["nodes"]) + 1 if not self.use_sub_token else len(graph["new_nodes"]) + 1] = 1
            adj_mat = np.ones((len(bert_input), len(bert_input)))
            if graph["funcs"] is not None:
                stage = cell_type(
                    graph["funcs"], nodes=graph["nodes"], header=graph["header"])
            else:
                stage = int(graph["stage"])
            neighbor_cells = graph["neighbor_cells"]
            id_ = graph["id"]
            markdown_input = ["[EPT]"] if graph["header"].strip() == "" else [
                t.lower() for t in graph["header"].split() if t]
            markdown_len = len(markdown_input)
            markdown_input = markdown_input + ["[PAD]"] * self.seq_len
            markdown_input = markdown_input[:self.seq_len]
            markdown_label = [self.markdown_vocab.word2idx.get(t, self.markdown_vocab.unk_index)
                              for t in markdown_input]
            # context_topic_vec = None if not self.context
            output = {"bert_input": bert_input,
                      "bert_label": bert_label,
                      "segment_label": segment_label,
                      # "is_next": is_next_label,
                      "adj_mat": adj_mat,
                      "seq_len": len(bert_input),
                      "stage": stage,
                      "neighbor_cells": neighbor_cells,
                      "id": id_,
                      "context_topic_vec": [self.neighbor_topic_dist.get(i, np.zeros(self.n_topics)) for i in neighbor_cells],
                      "markdown_label": markdown_label,
                      "markdown_len": markdown_len}
            return output
        pos_output = process_graph(self.graphs[item])
        neg_ids = random.sample(range(len(self.graphs)), self.n_neg)

        neg_output = [process_graph(
            self.graphs[i]) for i in neg_ids]

        return pos_output, neg_output
        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, graph):
        if self.use_sub_token:
            nodes = graph["new_nodes"]
            tokens = [self.vocab.word2idx.get(
                n, self.vocab.unk_index) for n in nodes]
            output_label = [self.vocab.word2idx.get(
                n, self.vocab.unk_index) for n in nodes]
            return tokens, output_label
            # raise NotImplementedError
        else:
            nodes = graph["nodes"]
            tokens = [n["value"] if "value" in n else n["type"] for n in nodes]
            # tokens = [self.vocab.word2idx.get(
            #     t, self.vocab.unk_index) for t in tokens]
        type_tokens = set([n["type"] for n in graph["nodes"]])
        # pdb.set_trace()

        output_label = []

        for i, token in enumerate(tokens):
            if token in type_tokens:
                tokens[i] = self.vocab.word2idx.get(
                    token, self.vocab.unk_index)
                output_label.append(self.vocab.pad_index)
                continue

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.word2idx.get(
                        token, self.vocab.unk_index)

                output_label.append(self.vocab.word2idx.get(
                    token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.word2idx.get(
                    token, self.vocab.unk_index)
                output_label.append(self.vocab.pad_index)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]

    def update_topic_dist(self, topic_vec, ids):

        # topic_dist, data["id"]
        for i, vec in zip(ids, topic_vec):
            self.neighbor_topic_dist[i] = vec
        # raise NotImplementedError


class TempDataset(Dataset):
    def __init__(self, graphs, vocab, seq_len, use_sub_token=False, encoding="utf-8", corpus_lines=None, on_memory=True, chunk_size=128, n_neg=5, n_topics=50):
        self.vocab = vocab
        # self.markdown_vocab = markdown_vocab
        # self.seq_len = seq_len if not use_sub_token else int(seq_len * 1.5)
        self.seq_len=seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines

        self.encoding = encoding
        self.chunk_size = chunk_size
        self.use_sub_token = use_sub_token
        self.n_neg = n_neg
        self.n_topics = n_topics

        self.neighbor_topic_dist = {
            g["id"]: np.zeros(n_topics) for g in graphs}
        # graphs = [g for g in graphs if len(g["nodes"]) + 1 <= seq_len]
        if use_sub_token:
            graphs = list(sorted(graphs, key = lambda x: len(x["new_nodes"])+len([t for t in ' '.join(x["annotation"]).split() if t])))
            # graphs = list(sorted(graphs, key=lambda x: len(x["new_nodes"])))
            # raise NotImplementedError
        else:
            graphs = list(sorted(graphs, key=lambda x: len(x["nodes"])))
        graphs = [graphs[i:i + chunk_size]
                  for i in range(0, len(graphs), chunk_size)]
        random.shuffle(graphs)
        graphs = list(itertools.chain.from_iterable(graphs))
        # for g in graphs:
        #     print()
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)
        # return self.corpus_lines

    def __getitem__(self, item):
        def process_graph(graph):
            # get code seq
            t1_label = self.get_code_seq(graph)
            # get markdown seq
            t2_label = self.get_markdown_seq(graph)
            # output 添加一个 neighbor topic 的属性
            bert_input = [self.vocab.sos_index]+t1_label+[self.vocab.sep_index]+t2_label+[self.vocab.eos_index]

            segment_label  = [1 ]*(2+len(t1_label))+[2]*(len(t2_label)+1)
            assert len(bert_input)==len(segment_label)

            bert_input = bert_input[:self.seq_len]
            segment_label = segment_label[:self.seq_len]

            adj_mat = np.ones((len(bert_input), len(bert_input)))

            if graph["funcs"] is not None:
                stage = cell_type(
                    graph["funcs"], nodes=graph["nodes"], header=graph["header"])
            else:
                stage = int(graph["stage"])

            output = {"bert_input":bert_input,
                      "segment_label":segment_label,
                      "adj_mat":adj_mat,
                      "stage":stage,
                      "seq_len":len(bert_input)}
            return output

        pos_output = process_graph(self.graphs[item])
        neg_ids = random.sample(range(len(self.graphs)), self.n_neg)

        neg_output = [process_graph(
            self.graphs[i]) for i in neg_ids]

        return pos_output, neg_output

    def get_code_seq(self, graph):
        if self.use_sub_token:
            nodes = graph["new_nodes"]
        else:
            raise NotImplementedError
        ids = [self.vocab.word2idx.get(n, self.vocab.unk_index) for n in nodes]

        return ids

    def get_markdown_seq(self, graph):
        markdown = graph["annotation"]
        markdown = " [SEP] ".join([l.lower() for l in markdown if l.strip()])

        tokens =[t for t in markdown.split() if t]
        ids = [self.vocab.word2idx.get(t, self.vocab.unk_index) for t in tokens]
        return ids




class SNAPDataset(Dataset):
    def __init__(self, graphs, vocab, seq_len, use_sub_token=False, encoding="utf-8", corpus_lines=None, on_memory=True, chunk_size=128, n_neg=5, n_topics=50, markdown = False):
        self.vocab = vocab

        self.seq_len=seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines

        self.encoding = encoding
        self.chunk_size = chunk_size
        self.use_sub_token = use_sub_token
        self.n_neg = n_neg
        self.n_topics = n_topics
        self.markdown = markdown
        self.neighbor_topic_dist = {
            g["id"]: np.zeros(n_topics) for g in graphs}
        # graphs = [g for g in graphs if len(g["nodes"]) + 1 <= seq_len]
        if use_sub_token:
            if markdown:
                graphs = list(sorted(graphs, key = lambda x: len(x["new_nodes"])+len([t for t in ' '.join(x["annotation"]).split() if t])))
            else:
                graphs = list(sorted(graphs, key=lambda x: len(x["new_nodes"])))
            # raise NotImplementedError
        else:
            graphs = list(sorted(graphs, key=lambda x: len(x["nodes"])))
        graphs = [graphs[i:i + chunk_size]
                  for i in range(0, len(graphs), chunk_size)]
        random.shuffle(graphs)
        graphs = list(itertools.chain.from_iterable(graphs))
        # for g in graphs:
        #     print()
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)
        # return self.corpus_lines

    def __getitem__(self, item):
        def process_graph(graph):
            # get code seq
            t1_label = self.get_code_seq(graph)
            # get markdown seq
            if self.markdown:
                t2_label =  self.get_markdown_seq(graph)
            # output 添加一个 neighbor topic 的属性

                bert_input = [self.vocab.sos_index]+t1_label+[self.vocab.sep_index]+t2_label+[self.vocab.eos_index]
                segment_label  = [1 ]*(2+len(t1_label))+[2]*(len(t2_label)+1)


            else:
                bert_input = [self.vocab.sos_index]+t1_label+[self.vocab.eos_index]
                segment_label  = [1 ]*(2+len(t1_label))
            assert len(bert_input)==len(segment_label)
            bert_input = bert_input[:self.seq_len]
            segment_label = segment_label[:self.seq_len]

            if self.use_sub_token:
                if self.markdown:
                    # raise NotImplementedError
                    adj_mat = np.zeros((3+len(t1_label)+len(t2_label),3+len(t1_label)+len(t2_label)))


                else:
                    adj_mat = np.zeros((len(graph["new_nodes"])+2, len(graph["new_nodes"])+2))
            else:
                raise NotImplementedError

            for i, n in enumerate(graph["nodes"]):
                if "children" not in n:
                    continue
                for c in n["children"]:
                    if self.use_sub_token:
                        # pdb.set_trace()
                        # print(graph["idx_map"][i])
                        # print(graph["idx_map"][c])
                        for ii in [z + 1 for z in graph["idx_map"][i]]:
                            for cc in [z + 1 for z in graph["idx_map"][c]]:
                                adj_mat[ii][cc]=1
                                adj_mat[cc][ii]=1
                        # adj_mat[[z + 1 for z in graph["idx_map"][i]],
                        #         [z + 1 for z in graph["idx_map"][c]]] = 1
                        # adj_mat[[z + 1 for z in graph["idx_map"][c]],
                        # [z + 1 for z in graph["idx_map"][i]]]=1
                    else:
                        raise NotImplementedError
                        adj_mat[i + 1][c + 1] = 1

            adj_mat[:,0]=1
            adj_mat[0,:]=1
            if self.markdown:
                adj_mat[2+len(t1_label):, :]=1
                adj_mat[:,2+len(t1_label):]=1
            adj_mat = adj_mat[:self.seq_len, :self.seq_len]
            if graph["funcs"] is not None:
                stage = cell_type(
                    graph["funcs"], nodes=graph["nodes"], header=graph["header"])


            else:
                stage = int(graph["stage"])

            output = {"bert_input":bert_input,
                      "segment_label":segment_label,
                      "adj_mat":adj_mat,
                      "stage":stage,
                      "seq_len":len(bert_input)}

            # pdb.set_trace()

            return output

        pos_output = process_graph(self.graphs[item])
        neg_ids = random.sample(range(len(self.graphs)), self.n_neg)
        # pdb.set_trace()
        neg_output = [process_graph(
            self.graphs[i]) for i in neg_ids]

        return pos_output, neg_output

    def get_code_seq(self, graph):
        if self.use_sub_token:
            nodes = graph["new_nodes"]
        else:
            raise NotImplementedError
        ids = [self.vocab.word2idx.get(n, self.vocab.unk_index) for n in nodes]

        return ids

    def get_markdown_seq(self, graph):
        markdown = graph["annotation"]
        markdown = " [SEP] ".join([l.lower() for l in markdown if l.strip()])

        tokens =[t for t in markdown.split() if t]
        ids = [self.vocab.word2idx.get(t, self.vocab.unk_index) for t in tokens]
        return ids


def my_collate(batch):
    def collate_graphs(graphs):
        # pdb.set_trace()
        seq_len = max([item["seq_len"] for item in graphs])
        # print(seq_len)
        bert_input = []
        bert_label = []
        segment_label = []
        adj_mat = []
        stages = []
        context_topic_vec = []
        ids = []
        markdown_label = []
        markdown_len = []
        for item in graphs:
            item["bert_input"] += [Vocab.pad_index] * \
                (seq_len - item["seq_len"])
            item["bert_label"] += [Vocab.pad_index] * \
                (seq_len - item["seq_len"])
            item["segment_label"] += [Vocab.pad_index] * \
                (seq_len - item["seq_len"])
            mat = np.zeros((seq_len, seq_len))
            mat[:item["adj_mat"].shape[0],
                :item["adj_mat"].shape[1]] = item["adj_mat"]
            # mat[:item["adj_mat"].shape[0], :item["adj_mat"].shape[1]
            #     ] = np.ones(item["adj_mat"].shape)
            bert_input.append(item["bert_input"])
            bert_label.append(item["bert_label"])
            segment_label.append(item["segment_label"])
            adj_mat.append(mat)
            stages.append(item["stage"])
            context_topic_vec.append(item["context_topic_vec"])
            ids.append(item["id"])
            markdown_label.append(item["markdown_label"])
            markdown_len.append(item["markdown_len"])
        # pdb.set_trace()

        return {"bert_input": torch.tensor(bert_input),
                "bert_label": torch.tensor(bert_label),
                "segment_label": torch.tensor(segment_label),
                "adj_mat": torch.tensor(adj_mat),
                "stage": torch.tensor(stages),
                "context_topic_vec": torch.tensor(context_topic_vec),
                "id": torch.tensor(ids),
                "markdown_label": torch.tensor(markdown_label),
                "markdown_len": torch.tensor(markdown_len)}
    pos_graphs = [item[0] for item in batch]
    neg_graphs = list(itertools.chain.from_iterable(
        [item[1] for item in batch]))
    # return batch
    return collate_graphs(pos_graphs), collate_graphs(neg_graphs)
    return batch
    # pdb.set_trace()




def temp_collate(batch):
    def collate_graphs(graphs):

        seq_len = max([item["seq_len"] for item in graphs])

        bert_input = []

        segment_label = []
        adj_mat = []
        stages = []

        for item in graphs:
            item["bert_input"] += [UnitedVocab.pad_index] * \
                (seq_len - item["seq_len"])

            item["segment_label"] += [UnitedVocab.pad_index] * \
                (seq_len - item["seq_len"])
            mat = np.zeros((seq_len, seq_len))
            mat[:item["adj_mat"].shape[0],
                :item["adj_mat"].shape[1]] = item["adj_mat"]

            bert_input.append(item["bert_input"])
            segment_label.append(item["segment_label"])
            adj_mat.append(mat)
            stages.append(item["stage"])


        return {"bert_input": torch.tensor(bert_input),
                "segment_label": torch.tensor(segment_label),
                "adj_mat": torch.tensor(adj_mat),
                "stage": torch.tensor(stages)}
    pos_graphs = [item[0] for item in batch]
    neg_graphs = list(itertools.chain.from_iterable(
        [item[1] for item in batch]))
    # return batch
    return collate_graphs(pos_graphs), collate_graphs(neg_graphs)
    # pdb.set_trace()



if __name__ == '__main__':
    data_reader = DataReader(
        "/homes/gws/gezhang/jupyter-notebook-analysis/graphs/cell_with_func_python23_1_27.txt", use_sub_token=True, seq_len=120, max_graph_num=10000)


    # vocab = UnitedVocab(data_reader.graphs, min_occur=10,
    #                     use_sub_token=True)

    vocab = SNAPVocab(data_reader.graphs, use_sub_token=True, min_occur=5)

    train_dataset = SNAPDataset(data_reader.graphs[:int(len(data_reader) * 0.8)], vocab, seq_len=120,
                                on_memory=True, n_neg=1, use_sub_token=True, n_topics=50)

    # a = train_dataset.__getitem__(1)
    # pdb.set_trace()
    # pdb.set_trace()
    # train_dataset.__getitem__(18)
    train_data_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=1, collate_fn=temp_collate)
    n_stages = 0
    n_valid_stages = 0
    # pdb.set_trace()
    i = 0
    for data in tqdm(train_data_loader):
        # print(i)
        # print()
        # print()
        pass
        # i+=1
        # a =  train_dataset.__getitem__(i)
        # pdb.set_trace()
        # print("hello")
        # pass
        # n_valid_stages += data[0]["stage"][data[0]["stage"] != 0].shape[0]
        # n_stages += data[0]["stage"].shape[0]
    pdb.set_trace()
    print(n_stages)
    print(n_valid_stages)
    print(n_valid_stages / n_stages)
    pdb.set_trace()
    for data in train_data_loader:

        pdb.set_trace()
        print('hello')
