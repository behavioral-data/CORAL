import numpy
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import pickle
import pdb
from dataset.prepare_data import cell_type
import torch.optim as optim

batch_size = 128
n_epochs = 50

graphs = []
with open('/projects/bdata/jupyter/gezhang_backup/jupyter-notebook-analysis/graphs/cell_with_func_python23_1_27.txt', 'r') as f:
    for l in tqdm(f):
        graphs.append(json.loads(l))

split_point = int(len(graphs) * 0.8)
lda_results = numpy.load(
    '/homes/gws/gezhang/jupyter-notebook-analysis/lda_results_3_9.npy', allow_pickle=True)
targets = [cell_type(g["funcs"], g["nodes"], g["header"]) for g in graphs]
targets = torch.LongTensor(targets).cuda()
# pdb.set_trace()


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 6)

    def forward(self, inputs):
        return self.linear(inputs)


'TODO'
# put model on GPU

# define model
classifier = Classifier()

# put model on CUDA
classifier.cuda()
# put lda results on CUDA
lda_results = torch.Tensor(lda_results)
lda_results = lda_results.cuda()
# pdb.set_trace()

# define criteria
criteria = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(classifier.parameters(), lr=0.05)
# train model
for epoch in range(n_epochs):
        # train
    all_loss = 0
    classifier.train()
    for i in tqdm(range(0, len(graphs[:split_point]), batch_size)):

        optimizer.zero_grad()
        dist_stage = classifier(lda_results[:split_point][i:i + batch_size])
        labels = targets[:split_point][i:i + batch_size]
        loss = criteria(dist_stage, labels)
        # print(loss.item())
        all_loss += loss.item()
        # back propogate gradient
        loss.backward()
        optimizer.step()
    print('train', all_loss / (len(graphs[:split_point]) // batch_size))
    # valid
    all_loss = 0
    classifier.eval()
    for i in tqdm(range(0, len(graphs[split_point:]), batch_size)):
        # optimizer.zero_grad()
        dist_stage = classifier(lda_results[split_point:][i:i + batch_size])
        labels = targets[split_point:][i:i + batch_size]
        loss = criteria(dist_stage, labels)
        # print(loss.item())
        all_loss += loss.item()
        # back propogate gradient
        # loss.backward()
        # optimizer.step()
    print('eval', all_loss / (len(graphs[split_point:]) // batch_size))


# test accuracy
test_graphs = []
with open('/projects/bdata/jupyter/gezhang_backup/jupyter-notebook-analysis/graphs/test.txt', 'r') as f:
    for l in tqdm(f):
        test_graphs.append(json.loads(l))
test_targets = [int(g["stage"]) for g in test_graphs]

test_lda_results = numpy.load(
    '/homes/gws/gezhang/jupyter-notebook-analysis/test_lda_results_3_9.npy', allow_pickle=True)
test_lda_results = torch.Tensor(test_lda_results)
test_lda_results = test_lda_results.cuda()

prediction = classifier(test_lda_results)
stages = prediction.max(1)[1]
count = 0
for i, j in zip(test_targets, stages):
    if i == j:
        count += 1
print(count / len(test_graphs))
pdb.set_trace()
