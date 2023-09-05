# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import argparse
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from datasets import RecWithContrastiveLearningDataset,RecWithDataset
from datasets import RecWithDataset
from trainers import CoSeRecTrainer
from models import SASRecModel, OfflineItemSimilarity, OnlineItemSimilarity
from GRU4Rec import GRU4RecModel
from NARM import NARMModel
from BERT4Rec import BERT4RecModel
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed, get_local_time

def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str) #Beauty Sports_and_Outdoors Toys_and_Games
    # parser.add_argument('--do_eval',default='0', action='store_true')
    parser.add_argument('--do_eval', type=int, default='0')
    parser.add_argument('--model_idx', default=0, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="1", help="gpu_id")

    #data augmentation args
    parser.add_argument('--noise_ratio', default=0.0, type=float, \
                        help="percentage of negative interactions in a sequence - robustness analysis")
    parser.add_argument('--training_data_ratio', default=1.0, type=float, \
                        help="percentage of training samples used for training - robustness analysis")
    parser.add_argument('--augment_threshold', default=4, type=int, \
                        help="control augmentations on short and long sequences.\
                        default:-1, means all augmentations types are allowed for all sequences.\
                        For sequence length < augment_threshold: Insert, and Substitute methods are allowed \
                        For sequence length > augment_threshold: Crop, Reorder, Substitute, and Mask \
                        are allowed.")
    parser.add_argument('--similarity_model_name', default='ItemCF_IUF', type=str, \
                        help="Method to generate item similarity score. choices: \
                        Random, ItemCF, ItemCF_IUF(Inverse user frequency), Item2Vec, LightGCN")
    parser.add_argument("--augmentation_warm_up_epoches", type=float, default=160, \
                        help="number of epochs to switch from \
                        memory-based similarity model to \
                        hybrid similarity model.")
    parser.add_argument('--base_augment_type', default='random', type=str, \
                        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, substitute, insert, random, \
                        combinatorial_enumerate (for multi-view).")
    parser.add_argument('--augment_type_for_short', default='SIM', type=str, \
                        help="data augmentation types for short sequences. Chosen from: \
                        SI, SIM, SIR, SIC, SIMR, SIMC, SIRC, SIMRC.")
    parser.add_argument("--tao", type=float, default=0.2, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.7, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.2, help="reorder ratio for reorder operator") 
    parser.add_argument("--substitute_rate", type=float, default=0.1, \
                        help="substitute ratio for substitute operator")
    parser.add_argument("--insert_rate", type=float, default=0.4, \
                        help="insert ratio for insert operator")
    parser.add_argument("--max_insert_num_per_pos", type=int, default=1, \
                        help="maximum insert items per position for insert operator - not studied")
    parser.add_argument('--temperature', default= 1.0, type=float,
                        help='softmax temperature (default:  1.0) - not studied.')
    parser.add_argument('--n_views', default=2, type=int, metavar='N',
                        help='Number of augmented data for each sequence - not studied.')

    # model args
    parser.add_argument("--model_name", default='ATSRec', type=str)
    parser.add_argument("--base_model_name",'-b', default='SASRec', type=str) # SASRec
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--loss_type', default="CE", type=str)
    parser.add_argument("--embedding_size", type=int, default=64, help="embedding_size of GRU4rec/NARM model")

    # BERT4Rec
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--n_heads', default=2, type=int)
    parser.add_argument("--inner_size", type=int, default=256, help="inner_size size of transformer model")
    #parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--attn_dropout_prob", type=float, default=0.5, help="attention dropout p")
    #parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12, help="attention dropout p")
    parser.add_argument("--mask_ratio", type=float, default=0.2, help="attention dropout p")
    #parser.add_argument('--loss_type', default="CE", type=str)
    #parser.add_argument("--initializer_range", type=float, default=0.02)
    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--cf_weight", type=float, default=0.1, \
                        help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, \
                        help="weight of contrastive learning task")

    #learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    parser.add_argument("--with_AT", default='Yes', type=str, help='whether add AT in base model')
    parser.add_argument("--adv_step", type=int, default=5, help="train step in adversarial training")
    parser.add_argument('--attack_train', type=str, required=False, default='pgd', help='choose a mode:pgd,FreeAT,fgsm')
    parser.add_argument('--epsilon', type=float, default=1.0, help='perturbation space of pgd attack')
    parser.add_argument('--eta', type=float, default=0.3, help='the step size for updating perturbation of pgd attack')

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + '.txt'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs(args.data_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    nowtime = get_local_time()
    args.nowtime = nowtime
    # save model args
    # args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args_str = f'{args.base_model_name}-{args.data_name}-{args.noise_ratio}-{args.training_data_ratio}-{args.nowtime}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')

    show_args_info(args)

    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    # save model
    #if args.noise_ratio == 0.0:
    #    checkpoint = args_str + '.pt'
    #    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    #else:
    #    args_str_noise = f'{args.base_model_name}-{args.data_name}-{0.0}-{args.training_data_ratio}'
    #    checkpoint_noise = args_str_noise + '.pt'
    #    args.checkpoint_path_noise = os.path.join(args.output_dir, checkpoint_noise)

    # -----------   pre-computation for item similarity   ------------ #
    args.similarity_model_path = os.path.join(args.data_dir,\
                            args.data_name+'_'+args.similarity_model_name+'_similarity.pkl')

    offline_similarity_model = OfflineItemSimilarity(data_file=args.data_file,
                            similarity_path=args.similarity_model_path,
                            model_name=args.similarity_model_name,
                            dataset_name=args.data_name)
    args.offline_similarity_model = offline_similarity_model

    # -----------   online based on shared item embedding for item similarity --------- #
    online_similarity_model = OnlineItemSimilarity(item_size=args.item_size)
    args.online_similarity_model = online_similarity_model

    # training data for node classification
    train_dataset = RecWithDataset(args,
                                    user_seq[:int(len(user_seq)*args.training_data_ratio)], \
                                    data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = RecWithDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = RecWithDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    if args.base_model_name == "SASRec":
        model = SASRecModel(args=args)
    elif args.base_model_name == "GRU4Rec":
        model = GRU4RecModel(args=args)
    elif args.base_model_name == "BERT4Rec":
        model = BERT4RecModel(args=args)
    else:
        model = NARMModel(args=args)
    trainer = CoSeRecTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)

    # if args.do_eval:
    #     trainer.args.train_matrix = test_rating_matrix
    #     trainer.load(args.checkpoint_path)
    #     print(f'Load model from {args.checkpoint_path} for test!')
    #     scores, result_info = trainer.test(0, full_sort=True)
    if args.do_eval == 1 and args.noise_ratio == 0:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    elif args.do_eval == 1 and args.noise_ratio != 0:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path_noise)
        print(f'Load model from {args.checkpoint_path_noise} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        print(f'Train ATSRec')
        early_stopping = EarlyStopping(args.checkpoint_path, patience=40, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        trainer.args.train_matrix = test_rating_matrix
        # Visualization of i
        # import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.decomposition import TruncatedSVD
        embedding_matrix = model.item_embeddings.weight[1:].cpu().detach().numpy()
        svd = TruncatedSVD(n_components=2)
        svd.fit(embedding_matrix)
        comp_tr = np.transpose(svd.components_)
        proj = np.dot(embedding_matrix, comp_tr)

        cnt = {}
        for j in user_seq:
            for i in j:
                if i != 0:
                    if i in cnt:
                        cnt[i] += 1
                    else:
                        cnt[i] = 1
        freq = np.zeros(embedding_matrix.shape[0])
        for i in cnt:
            freq[i - 1] = cnt[i]
        # freq /= freq.max()
        sns.set(style='darkgrid') # whitegrid darkgrid dark, white, ticks
        sns.set_context("notebook", font_scale=1.7, rc={"lines.linewidth": 3, 'lines.markersize': 10})
        plt.figure(figsize=(6, 4.5))
        plt.scatter(proj[:, 0], proj[:, 1], s=1, c=freq, cmap='cividis_r') # viridis_r  coolwarm_r cividis_r RdBu_r
        plt.colorbar()
        plt.xlim(-2, 2)  # plt.xlim(-4, 2)
        plt.ylim(-2, 2)  # plt.xlim(-2, 2)
        # plt.axis('square')
        # plt.show()
        #plt.savefig(args.output_dir + args.base_model_name + '-' + args.data_name + '/svs.pdf', format='pdf', transparent=False, bbox_inches='tight')
        plt.savefig(args.output_dir + args.base_model_name + '-' + args.data_name + args.nowtime + '-' + 'latexsvs.pdf', format='pdf',
                    transparent=False, bbox_inches='tight')
        from scipy.linalg import svdvals
        svs = svdvals(embedding_matrix)
        svs /= svs.max()
        # np.save(args_str + '/sv.npy', svs)
        np.save(args.output_dir + args.base_model_name + '-' + args.data_name + 'sv1.npy', svs)

        sns.set(style='darkgrid')
        sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 3, 'lines.markersize': 20})
        plt.figure(figsize=(6, 4.5))
        plt.plot(svs)
        # plt.show()
        plt.savefig(args.output_dir + args.base_model_name + args.data_name + 'sv2.pdf', format='pdf', transparent=False, bbox_inches='tight')
        # exit()
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()
