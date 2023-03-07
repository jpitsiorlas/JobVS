import os
import sys
import glob
import utils
import logging
import argparse
import os.path as osp

class ArgsInit(object):
    """_summary_

    Args:
        object (_type_): _description_
    """
    def __init__(self):
        parser = argparse.ArgumentParser(description='DeeperGCN')
        
        # -------------------------------------------------------
        # -----------------     DATA     ------------------------
        parser.add_argument('--data_path', type=str,
                            default='/home/pitsiorl/Shiny_Icarus/data/train',
                            help='Path to data')
        parser.add_argument('--json_data_file', type=str,
                            default='all_data_v1.json',
                            help='Data in json file')
        
        # -------------------------------------------------------
        # -----------------     DATASET     ---------------------
        parser.add_argument('--dataset_path', type=str,
                            default='/home/pitsiorl/Shiny_Icarus/data/dataset',
                            help='Path to dataset organized annotations')
        parser.add_argument('--dataset_version', type=str,
                            default='original',
                            help='Dataset version')
        parser.add_argument('--fold', type=str,
                            default='1',
                            help='Crossvalidation fold')
        parser.add_argument('--use_normal_dataset', action='store_true',
                            help='use monai Dataset class')
        parser.add_argument('--RandFlipd_prob', type=float,
                            default=0.2,
                            help='RandFlipd aug probability')
        parser.add_argument('--RandRotate90d_prob', type=float,
                            default=0.2, 
                            help='RandRotate90d aug probability')
        parser.add_argument('--RandScaleIntensityd_prob', type=float,
                            default=0.1, 
                            help='RandScaleIntensityd aug probability')
        parser.add_argument('--RandShiftIntensityd_prob', type=float,
                            default=0.1, 
                            help='RandShiftIntensityd aug probability')
                
        # -------------------------------------------------------
        # ----------------     TRAINING     ---------------------
        parser.add_argument('--distributed', action='store_true',
                            help='start distributed training')
        parser.add_argument('--world_size', type=int,
                            default=1, 
                            help='number of nodes for distributed training')
        parser.add_argument('--rank',  type=int,
                            default=0,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', type=str,
                            default='tcp://127.0.0.1:23456',
                            help='distributed url')
        parser.add_argument('--dist-backend', type=str,
                            default='nccl',
                            help='distributed backend')
        parser.add_argument('--workers', type=int,
                            default=8,
                            help='number of workers')
        parser.add_argument('--max_epochs', type=int, 
                            default=5000,
                            help='max number of training epochs')
        parser.add_argument('--batch_size', type=int,
                            default=1,
                            help='number of batch size')
        parser.add_argument('--sw_batch_size', type=int,
                            default=1,
                            help='number of sliding window batch size')
        parser.add_argument('--optim_lr', type=float,
                            default=1e-4,
                            help='optimization learning rate')
        parser.add_argument('--optim_name', type=str,
                            default='adamw',
                            help='optimization algorithm')
        parser.add_argument('--reg_weight', type=float,
                            default=1e-5, 
                            help='regularization weight')
        parser.add_argument('--momentum', type=float,
                            default=0.99,
                            help='momentum')
        parser.add_argument('--lrschedule', type=str,
                            default='warmup_cosine',
                            help='type of learning rate scheduler')
        parser.add_argument('--warmup_epochs', type=int,
                            default=50,
                            help='number of warmup epochs')
        parser.add_argument('--smooth_dr', type=float,
                            default=1e-6,
                            help='constant added to dice denominator to avoid nan')
        parser.add_argument('--smooth_nr', type=float,
                            default=0.0, 
                            help='constant added to dice numerator to avoid zero')
        parser.add_argument('--resume_jit', action='store_true',
                            help='resume training from pretrained torchscript checkpoint')
        parser.add_argument('--noamp', action='store_true',
                            help='do NOT use amp for training')
        parser.add_argument('--seed', type=int, default=1,
                            help='Seed for numpy and torch')
        # -------------------------------------------------------
        # --------------     VALIDATION     ---------------------
        parser.add_argument('--val_every', type=int,
                            default=100, 
                            help='validation frequency')
        parser.add_argument('--infer_overlap', type=float,
                            default=0.5,
                            help='sliding window inference overlap')
        
        # -------------------------------------------------------
        # -------------------     MODEL     ---------------------
        parser.add_argument('--model_name', type=str, 
                            default='unetr',
                            help='model name')
        parser.add_argument('--pos_embed',  type=str,
                            default='perceptron',
                            help='type of position embedding')
        parser.add_argument('--norm_name', type=str,
                            default='instance',
                            help='normalization layer type in decoder')
        parser.add_argument('--num_heads', type=int,
                            default=12, 
                            help='number of attention heads in ViT encoder')
        parser.add_argument('--mlp_dim', type=int,
                            default=3072,
                            help='mlp dimention in ViT encoder')
        parser.add_argument('--hidden_size', type=int,
                            default=768,
                            help='hidden size dimention in ViT encoder')
        parser.add_argument('--feature_size', type=int,
                            default=16,
                            help='feature size dimention')
        parser.add_argument('--in_channels', type=int,
                            default=1, 
                            help='number of input channels')
        parser.add_argument('--out_channels', type=int,
                            default=14,
                            help='number of output channels')
        parser.add_argument('--res_block', action='store_true',
                            help='use residual blocks')
        parser.add_argument('--conv_block', action='store_true',
                            help='use conv blocks')
        parser.add_argument('--a_min', type=float,
                            default=-175.0,
                            help='a_min in ScaleIntensityRanged')
        parser.add_argument('--a_max', type=float,
                            default=250.0,
                            help='a_max in ScaleIntensityRanged')
        parser.add_argument('--b_min', type=float,
                            default=0.0, 
                            help='b_min in ScaleIntensityRanged')
        parser.add_argument('--b_max', type=float,
                            default=1.0, 
                            help='b_max in ScaleIntensityRanged')
        parser.add_argument('--space_x', type=float,
                            default=1.0, 
                            help='spacing in x direction')
        parser.add_argument('--space_y', type=float,
                            default=1.0,
                            help='spacing in y direction')
        parser.add_argument('--space_z', type=float,
                            default=1.0, 
                            help='spacing in z direction')
        parser.add_argument('--roi_x', type=int,
                            default=96,
                            help='roi size in x direction')
        parser.add_argument('--roi_y', type=int,
                            default=96, 
                            help='roi size in y direction')
        parser.add_argument('--roi_z', type=int,
                            default=96,
                            help='roi size in z direction')
        parser.add_argument('--dropout_rate', type=float,
                            default=0.0, 
                            help='dropout rate')
        
        # -------------------------------------------------------
        # --------------     CHECKPOINT     ---------------------
        parser.add_argument('--checkpoint', default=None,
                            help='start training from saved checkpoint')
        parser.add_argument('--save_checkpoint', action='store_true',
                            help='save checkpoint during training')
        parser.add_argument('--resume_ckpt', action='store_true',
                            help='resume training from pretrained checkpoint')
        parser.add_argument('--pretrained_dir', type=str,
                            default='./pretrained_models/', 
                            help='pretrained checkpoint directory')
        
        # -------------------------------------------------------
        # -------------------     OUTPUT     --------------------   
        parser.add_argument('--logdir', type=str,
                            default='test', 
                            help='directory to save the tensorboard logs')
        
        parser.add_argument('--data_dir', type=str,
                            default='/dataset/dataset0/', 
                            help='dataset directory')
        parser.add_argument('--json_list', type=str, 
                            default='dataset_0.json',
                            help='dataset json file')
        parser.add_argument('--pretrained_model_name', type=str,
                            default='UNETR_model_best_acc.pth',
                            help='pretrained model name')
        
        self.args = parser.parse_args()
        # # dataset
        
        # parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
        #                     help='dataset name (default: ogbg-molhiv)')
        # parser.add_argument('--num_workers', type=int, default=0,
        #                     help='number of workers (default: 0)')
        # parser.add_argument('--batch_size', type=int, default=5120,
        #                     help='input batch size for training (default: 5120)')
        # parser.add_argument('--feature', type=str, default='full',
        #                     help='two options: full or simple')
        # parser.add_argument('--add_virtual_node', action='store_true')
        # # training & eval settings
        # parser.add_argument('--use_gpu', action='store_true')
        # parser.add_argument('--device', type=int, default=0,
        #                     help='which gpu to use if any (default: 0)')
        # parser.add_argument('--epochs', type=int, default=20,
        #                     help='number of epochs to train (default: 300)')
        # parser.add_argument('--lr', type=float, default=5e-5,
        #                     help='learning rate set for optimizer.')
        # parser.add_argument('--dropout', type=float, default=0.2)
        # # model
        # parser.add_argument('--num_layers', type=int, default=20,
        #                     help='the number of layers of the networks')
        # parser.add_argument('--mlp_layers', type=int, default=3,
        #                     help='the number of layers of mlp in conv')
        # parser.add_argument('--hidden_channels', type=int, default=128,
        #                     help='the dimension of embeddings of nodes and edges')
        # parser.add_argument('--block', default='res+', type=str,
        #                     help='graph backbone block type {res+, res, dense, plain}')
        # parser.add_argument('--conv', type=str, default='gen',
        #                     help='the type of GCNs')
        # parser.add_argument('--gcn_aggr', type=str, default='softmax',
        #                     help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
        # parser.add_argument('--norm', type=str, default='batch',
        #                     help='the type of normalization layer')
        # parser.add_argument('--num_tasks', type=int, default=1,
        #                     help='the number of prediction tasks')
       

    def return_args(self):
        
        self.args.original_data_path = osp.join(self.args.data_path, 'TrainVolumes')
        self.args.anns_data_path = osp.join(self.args.data_path, 'TrainManualLabels')
        self.args.smile_data_path = osp.join(self.args.data_path, 'smile')

        self.args.file_path = osp.join(self.args.data_path, 'stats.csv')

        # Modify dataset path according to data version.
        self.args.dataset_path = osp.join(self.args.dataset_path, self.args.dataset_version)
        utils.makedir(self.args.dataset_path)
        return self.args

        # self.args.save = '/{}/Fold{}'.format(self.args.save, str(self.args.cross_val))
        # '''
        # self.args.save = '{}-B_{}-C_{}-L_{}-F_{}-DP_{}' \
        #             '-GA_{}-T_{}-LT_{}-P_{}-LP_{}' \
        #             '-MN_{}-LS_{}'.format(self.args.save, self.args.block, self.args.conv,
        #                                   self.args.num_layers, self.args.hidden_channels,
        #                                   self.args.dropout, self.args.gcn_aggr,
        #                                   self.args.t, self.args.learn_t, self.args.p, self.args.learn_p,
        #                                   self.args.msg_norm, self.args.learn_msg_scale)'''

        # self.args.save = 'log/{}'.format(self.args.save)
        # self.args.model_save_path = os.path.join(self.args.save, self.args.model_save_path)
        # create_exp_dir(self.args.save, scripts_to_save=glob.glob('*.py'))
        # log_format = '%(asctime)s %(message)s'
        # logging.basicConfig(stream=sys.stdout,
        #                     level=logging.INFO,
        #                     format=log_format,
        #                     datefmt='%m/%d %I:%M:%S %p')
        # fh = logging.FileHandler(os.path.join(self.args.save, 'log.txt'))
        # fh.setFormatter(logging.Formatter(log_format))
        # logging.getLogger().addHandler(fh)
