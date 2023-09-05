# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from torch.utils.data import DataLoader, RandomSampler
#from datasets import RecWithContrastiveLearningDataset
from datasets import RecWithDataset
from modules import NCELoss, NTXent
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr
# from PGD import PGD
from attack_train import FGSM,PGD,FreeAT,FGM
class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        self.online_similarity_model = args.online_similarity_model

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        #projection head for contrastive learn task
        self.projection = nn.Sequential(nn.Linear(self.args.max_seq_length*self.args.hidden_size, \
                                        512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True), 
                                        nn.Linear(512, self.args.hidden_size, bias=True))
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        # self.cf_criterion = NTXent()
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)
        
    def __refresh_training_dataset(self, item_embeddings):
        """
        use for updating item embedding
        """
        user_seq, _, _, _ = get_user_seqs(self.args.data_file)
        self.args.online_similarity_model.update_embedding_matrix(item_embeddings)
        # training data for node classification
        train_dataset = RecWithDataset(self.args, user_seq[:int(len(user_seq)*self.args.training_data_ratio)],
                                        data_type='train', similarity_model_type='hybrid')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        return train_dataloader
        
    def train(self, epoch):
        # start to use online item similarity
        if epoch > self.args.augmentation_warm_up_epoches:
            print("refresh dataset with updated item embedding")
            self.train_dataloader = self.__refresh_training_dataset(self.model.item_embeddings)
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score0(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        #for k in [5, 10, 15, 20]:
        for k in [5, 10, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "HIT@10": '{:.4f}'.format(recall[1]),"HIT@20": '{:.4f}'.format(recall[2]),
            "NDCG@5": '{:.4f}'.format(ndcg[0]),"NDCG@10": '{:.4f}'.format(ndcg[1]),"NDCG@20": '{:.4f}'.format(ndcg[2])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')

        return [recall[0], recall[1], recall[2],ndcg[0],ndcg[1], ndcg[2]], str(post_fix)
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def item_seq_len(self,input_ids):
        a = []
        for i in input_ids:
            a.append(len(torch.nonzero(i)))
        a = torch.tensor(a)
        return a
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        #print("seq_out==", seq_out.size())  # seq_emb== 12800
        pos_emb = self.model.item_embeddings(pos_ids)
        #print("pos_emb==", pos_emb.size())  # seq_emb== 12800
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]

        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.args.max_seq_length).float() # [batch*seq_len]

        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class CoSeRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, 
                 args):
        super(CoSeRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, 
            args
        )

    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''

        cl_batch = torch.cat(inputs, dim=0)  #torch.Size([512, 50])
        cl_batch = cl_batch.to(self.device)
        #cl_sequence_output = self.model.transformer_encoder(cl_batch)
        cl_sequence_output = self.model.forward(cl_batch) # torch.Size([512, 50, 64])
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1) #  torch.Size([512, 3200])
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0]//2 # 256
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)
        #print("cl_output_slice==",cl_output_slice[0].size()) #torch.Size([256, 3200])

        cl_loss = self.cf_criterion(cl_output_slice[0], 
                                cl_output_slice[1])
        return cl_loss

    def iteration(self, epoch, dataloader, full_sort=True, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        if train:
            if self.args.attack_train == 'fgsm':
                fgsm = FGSM(model=self.model,eps=self.args.epsilon)
            elif self.args.attack_train == 'FreeAT':
                free_at = FreeAT(model=self.model)
            elif self.args.attack_train == 'fgm':
                fgm = FGM(model=self.model)
            elif self.args.attack_train == 'pgd':
                pgd = PGD(model=self.model,eps=self.args.epsilon,eta=self.args.eta)
            self.model.train()
            self.model.train()
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}") # 140
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader)) #total参数设置进度条的总长度

            for i, (rec_batch, cl_batches) in rec_cf_data_iter:
                '''
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                '''

                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch


                # ---------- recommendation task ---------------#
                sequence_output = self.model.forward(input_ids)
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)

                joint_loss = self.args.rec_weight * rec_loss # rec_weight=1
                joint_loss.backward()
                if self.args.with_AT == "Yes":
                    if self.args.attack_train == 'fgsm':
                        fgsm.attack()
                        sequence_output = self.model.forward(input_ids)
                        rec_loss_adv = self.cross_entropy(sequence_output, target_pos, target_neg)
                        joint_loss_adv =rec_loss_adv
                        joint_loss_adv.backward()
                        # rec_loss_adv.backward()
                        fgsm.restore()

                    elif self.args.attack_train == 'pgd':
                        # print("11111",111)
                        pgd_k = self.args.adv_step
                        pgd.backup_grad()
                        for _t in range(pgd_k):
                            pgd.attack(is_first_attack=(_t == 0))
                            if _t != pgd_k - 1:
                                self.model.zero_grad()
                            else:
                                pgd.restore_grad()
                            # sequence_output = self.model.forward(input_ids)
                            # rec_loss_adv = self.cross_entropy(sequence_output, target_pos, target_neg)
                            # rec_loss_adv.backward()
                            sequence_output = self.model.forward(input_ids)
                            rec_loss_adv = self.cross_entropy(sequence_output, target_pos, target_neg)
                            joint_loss_adv = rec_loss_adv
                            joint_loss_adv.backward()
                        pgd.restore()

                    elif self.args.attack_train == 'FreeAT':
                        m = 4
                        free_at.backup_grad()
                        for _t in range(m):
                            free_at.attack(is_first_attack=(_t == 0))
                            if _t != m - 1:
                                self.model.zero_grad()
                            else:
                                free_at.restore_grad()
                            # sequence_output = self.model.forward(input_ids)
                            # rec_loss_adv = self.cross_entropy(sequence_output, target_pos, target_neg)
                            # rec_loss_adv.backward()
                            sequence_output = self.model.forward(input_ids)
                            rec_loss_adv = self.cross_entropy(sequence_output, target_pos, target_neg)
                            joint_loss_adv = rec_loss_adv
                            joint_loss_adv.backward()
                        free_at.restore()

                    elif self.args.attack_train == 'fgm':
                        fgm.attack()
                        # sequence_output = self.model.forward(input_ids)
                        # rec_loss_adv = self.cross_entropy(sequence_output, target_pos, target_neg)
                        # rec_loss_adv.backward()
                        sequence_output = self.model.forward(input_ids)
                        rec_loss_adv = self.cross_entropy(sequence_output, target_pos, target_neg)
                        joint_loss_adv = rec_loss_adv
                        joint_loss_adv.backward()
                        fgm.restore()
                    self.optim.step()
                    self.model.zero_grad()
                else:
                    self.optim.step()
                    self.model.zero_grad()

                rec_avg_loss += rec_loss.item() # t.item()将Tensor变量转换为python标量（int float等）

                # for i, cl_loss in enumerate(cl_losses):
                #     cl_individual_avg_losses[i] += cl_loss.item()
                #     cl_sum_avg_loss += cl_loss.item()
                # joint_avg_loss += joint_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_cf_data_iter)),
               # "joint_avg_loss": '{:.4f}'.format(joint_avg_loss / len(rec_cf_data_iter)),
               # "cl_avg_loss": '{:.4f}'.format(cl_sum_avg_loss / (len(rec_cf_data_iter)*self.total_augmentaion_pairs)),
            }
            # for i, cl_individual_avg_loss in enumerate(cl_individual_avg_losses):
            #     post_fix['cl_pair_'+str(i)+'_loss'] = '{:.4f}'.format(cl_individual_avg_loss / len(rec_cf_data_iter))

            if (epoch + 1) % self.args.log_freq == 0: # args.log_freq=1
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else: #验证和测试集
            rec_data_iter = tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch), # Recommendation EP_test:0
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model.forward(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    """
                    今天遇到一个新的函数np.argpartition（）， 它是排序函数里的一个，返回的是索引。
                    它的思想是：根据一个数值x，把数组中的元素划分成两半，使得index前面的元素都不大于x，index后面的元素都不小于x。
                    np.argpartition认会从小到大"排序",此排序并非真正的排序,而是指定的元素索引落入正确的值时就停止排序.
                    test = np.array([5, 1, 2, 7, 3, 6, 4]) print(test[np.argpartition(test, 4)])
                    [3 4 2 1 5 6 7] #记为res
                    """
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                return self.get_sample_scores(epoch, pred_list)


