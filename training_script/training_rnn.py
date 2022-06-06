"""
This file defines the training pipeline and do the training
"""

import torch
from dataset.imi_dataset import ImiDataset, ImiDatasetLabelCount
from models.vision_encoders import make_vision_encoder,make_pos_encoder
from models.imi_models import *
from engine.engine import *
from models.actors import *
from torch.utils.data import DataLoader
from boilerplate import *
import pandas as pd

def main(args):

    # get the csv file storing all episodes
    train_csv = pd.read_csv(args.train_csv)
    val_csv = pd.read_csv(args.val_csv)
    # count training set len and val set len
    if args.num_episode is None:
        train_num_episode = len(train_csv)
        val_num_episode = len(val_csv)
    else:
        train_num_episode = args.num_episode
        val_num_episode = args.num_episode

    
    train_label_set = torch.utils.data.ConcatDataset([ImiDatasetLabelCount(args.train_csv, args, i, args.data_folder) for i in range(train_num_episode)])
    # define dataset and data loaders
    train_set = torch.utils.data.ConcatDataset(
        [ImiDataset(args.train_csv, args, i, args.data_folder) for i in range(train_num_episode)])
    val_set = torch.utils.data.ConcatDataset(
        [ImiDataset(args.train_csv, args, i, args.data_folder, False) for i in
         range(val_num_episode)])
        
    # create weighted sampler to balance samples
    train_label = []
    for keyboard in train_label_set:
        train_label.append(keyboard)
    class_sample_count = np.zeros(27)
    for t in np.unique(train_label):
        class_sample_count[t] = len(np.where(train_label == t)[0])
    weight = 1. / (class_sample_count + 1e-5)
    samples_weight = np.array([weight[t] for t in train_label])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_loader = DataLoader(train_set, args.batch_size, num_workers=4, sampler=sampler)
    val_loader = DataLoader(val_set, 1, num_workers=4)

    # define vision encoder and imitation network and combine them in the actor
    vision_encoder = make_vision_encoder()
    pos_encoder = make_pos_encoder()
    imi_model = rnn_imi_networks()
    actor = rnn_robotActor(vision_encoder, pos_encoder, imi_model,args)

    # # define optimizer and lr scheduler
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(actor.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.period, gamma=args.gamma)
    # save config
    exp_dir = save_config(args)
    # pl stuff
    pl_module = RNN_RobotLearning(actor, optimizer, train_loader, val_loader, scheduler, args)
    start_training(args, exp_dir, pl_module)

if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="config/imi_config.yaml")
    p.add("--batch_size", default=16,type=int)
    p.add("--lr", default=0.001, type=float)
    p.add("--gamma", default=0.9, type=float)
    p.add("--period", default=3,type=int)
    p.add("--epochs", default=500,type=int)
    p.add("--num_episode", default=None, type=int)
    p.add("--resume", default=None)
    p.add("--num_workers", default=4, type=int) # defult used to be 8
    p.add("--num_camera", default=1, type=int)
    p.add("--use_convnext", default=False)
    # imi_stuff
    p.add("--freeze_till", required = True, type=int)
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="data/data_0331/test_recordings")
    p.add("--resized_height", required=True, type=int)
    p.add("--resized_width", required=True, type=int)
    p.add("--crop_per", required=True, type=float)



    args = p.parse_args()
    main(args)