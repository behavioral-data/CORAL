# coding=utf-8
# created by Ge Zhang, Jan 20, 2020
#
# annotation test file


import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import BERT, CORAL
# from .optim_schedule import ScheduledOptim

import tqdm
import pdb
torch.manual_seed(0)


def my_loss(reconstructed_pos, origin_pos, origin_neg):
    duplicate = int(origin_neg.shape[0] / reconstructed_pos.shape[0])

    hid_size = origin_neg.shape[-1]

    pos_sim = torch.bmm(reconstructed_pos.unsqueeze(
        1), origin_pos.unsqueeze(2)).repeat(1, duplicate, 1).view(-1)
    neg_sim = torch.bmm(reconstructed_pos.repeat(
        1, duplicate).view(-1, hid_size).unsqueeze(1), origin_neg.unsqueeze(2)).view(-1)
    diff = neg_sim - pos_sim + 1

    diff = torch.max(diff, torch.zeros_like(diff))
    loss = torch.sum(diff)
    return loss


class CORALTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, pad_index=0, loss_lambda=1, model_path=None, n_topics=50, weak_supervise=False, context=False, markdown=False, hinge_loss_start_point=20, entropy_start_point=30):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        :param context: use information from neighbor cells
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        self.loss_lambda = loss_lambda
        self.n_topics = n_topics
        self.weak_supervise = weak_supervise
        self.context = context
        self.markdown = markdown
        self.hinge_loss_start_point = hinge_loss_start_point
        self.entropy_start_point = entropy_start_point
        cuda_condition = torch.cuda.is_available() and with_cuda

        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = CORAL(bert, vocab_size, n_topics=n_topics,
                           weak_supervise=weak_supervise, context=context, markdown=markdown).to(self.device)
        print(model_path)
        if model_path:
            state_dict = torch.load(model_path)["model_state_dict"]
            # pdb.set_trace()
            model_dict = self.model.state_dict()
            model_dict.update(state_dict)
            self.model.load_state_dict(state_dict)
            # self.model.load_state_dict(
            #     torch.load(model_path)["model_state_dict"])

            # Distributed GPU training if CUDA can detect more than 1 GPU
        # pdb.set_trace()
        if with_cuda and torch.cuda.device_count() > 1:
            # pdb.set_trace()
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        # pdb.set_trace()
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.pad_index = pad_index
        # Setting the Adam optimizer with hyper-param
        # self.optim = Adam(self.model.parameters(), lr=lr,
        #                   betas=betas, weight_decay=weight_decay)
        # self.optim_schedule = ScheduledOptim(
        #     self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)
        self.optim = SGD(self.model.parameters(), lr=lr, momentum=0.9)
        # Using Negative Log Likelihood Loss function for predicting the masked_token
        # self.criterion = nn.NLLLoss(ignore_index=self.pad_index)
        self.best_loss = None
        self.updated = False
        self.log_freq = log_freq
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

        print("Total Parameters:", sum([p.nelement()
                                        for p in self.model.parameters()]))

    def train(self, epoch):
        self.model.train()
        # self.optim.zero_grad()

        return self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            loss = self.iteration(epoch, self.test_data, train=False)
        return loss

    def api(self, data_loader=None):
        self.model.eval()

        # str_code = "train" if train else "test"
        if not data_loader:
            data_loader = self.test_data

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              # desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        # for (i, data), (ni, ndata) in data_iter, neg_data_iter:
        phases = []
        stages = []
        stage_vecs = []
        with torch.no_grad():
            for i, item in data_iter:
                data = item[0]
                ndata = item[1]
                data = {key: value.to(self.device)
                        for key, value in data.items()}
                ndata = {key: value.to(self.device)
                         for key, value in ndata.items()}

                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device)
                        for key, value in data.items()}
                ndata = {key: value.to(self.device)
                         for key, value in ndata.items()}
                # pdb.set_trace()
                # 1. forward the next_sentence_prediction and masked_lm model
                # pdb.set_trace()
                reconstructed_vec, graph_vec, origin_neg, topic_dist, stage_vec = self.model.forward(
                    data["bert_input"], ndata["bert_input"], data["segment_label"], ndata["segment_label"], data["adj_mat"], ndata["adj_mat"], train=False)
                # data_loader.dataset.update_topic_dist(topic_dist, data["id"])

                # phases += torch.max(topic_dist, 1)[-1].tolist()
                # print(torch.max(stage_vec, 1)[-1].tolist())
                stages += torch.max(stage_vec, 1)[-1].tolist()
                stage_vecs += stage_vec.tolist()
                # pdb.set_trace()
        return stages, stage_vecs

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0

        # def calculate_iter(data):

        for i, item in data_iter:
            # if train:
            #     self.optim.zero_grad()
            data = item[0]
            ndata = item[1]

            data = {key: value.to(self.device) for key, value in data.items()}
            ndata = {key: value.to(self.device)
                     for key, value in ndata.items()}

            reconstructed_vec, graph_vec, origin_neg, topic_dist, stage_vec = self.model.forward(
                data["bert_input"], ndata["bert_input"], data["segment_label"], ndata["segment_label"], data["adj_mat"], ndata["adj_mat"], train=train)

            bs, _ = reconstructed_vec.shape
            nbs, _ = origin_neg.shape
            duplicate = int(nbs / bs)

            hinge_loss = my_loss(reconstructed_vec, graph_vec, origin_neg)
            weight_loss = torch.norm(torch.mm(
                self.model.reconstruction.weight.T, self.model.reconstruction.weight) - torch.eye(self.n_topics).cuda())
            c_entropy = self.cross_entropy(stage_vec, data['stage'])
            entropy = -1 * (F.softmax(stage_vec, dim=1) *
                            F.log_softmax(stage_vec, dim=1)).sum()

            if epoch < self.hinge_loss_start_point:
                loss = c_entropy

            elif epoch < self.entropy_start_point:
                loss = c_entropy + self.loss_lambda * weight_loss + hinge_loss
            else:
                loss = c_entropy + entropy + self.loss_lambda * weight_loss + hinge_loss

            if epoch == self.hinge_loss_start_point:
                self.optim = SGD(self.model.parameters(),
                                 lr=0.00001, momentum=0.9)

            # 3. backward and optimization only in train

            if train:
                self.optim.zero_grad()
                loss.backward()
                # self.optim.step_and_update_lr()
                self.optim.step()

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                # "avg_acc": total_correct / total_element * 100,
                "loss": loss.item(),
                "cross_entropy": c_entropy.item(),
                "entropy": entropy.item(),
                "hinge_loss": hinge_loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" %
              (epoch, str_code), avg_loss / len(data_iter))
        return avg_loss / len(data_iter)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict()
        }, output_path)
        print("EP:%d Model Saved on:" % epoch, output_path)

        return output_path
