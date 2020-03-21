# coding=utf-8
# super simple baseline

from torch.utils.data import DataLoader
from trainer import BERTTrainer, ReconstructionBERTTrainer
from dataset import BERTDataset, WordVocab, DataReader, Vocab, my_collate, CustomBERTDataset
import pdb
import random

labeled_data_reader = DataReader(
    '/homes/gws/gezhang/jupyter-notebook-analysis/graphs/test_cells_1_11.txt', use_sub_token=True)

gold_truth = [int(g["stage"]) for g in labeled_data_reader.graphs]
prediction = []
for g in labeled_data_reader.graphs:
    nodes = g["new_nodes"]
    # pdb.set_trace()
    if g["funcs"] is None:
        prediction.append(random.choice([1, 2, 3, 4, 5]))
    elif any([f.startswith('pandas') for f in g["funcs"]]):

        prediction.append(1)
    elif any([f.startswith('sklearn') for f in g["funcs"]]):
        prediction.append(3)
    elif any([f.startswith('seaborn') or f.startswith('matplotlib') for f in g["funcs"]]):
        prediction.append(2)

    else:
        prediction.append(random.choice([1, 2, 3, 4, 5]))
correct = 0
for i, j in zip(prediction, gold_truth):
    if i == j:
        correct += 1
print(correct / len(gold_truth))
pdb.set_trace()
