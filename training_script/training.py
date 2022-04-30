"""
This file defines the training pipeline and do the training
"""

import torch
from dataset.imi_dataset import ImiDataset
from models.vision_encoders import make_vision_encoder
from models.imi_models import Imi_networks
from engine.engine import RobotLearning
from models.actors import robotActor
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

    # define dataset and data loaders
    train_set = torch.utils.data.ConcatDataset(
        [ImiDataset(args.train_csv, args, i, args.data_folder) for i in range(train_num_episode)])
    val_set = torch.utils.data.ConcatDataset(
        [ImiDataset(args.val_csv, args, i, args.data_folder, False) for i in
         range(val_num_episode)])
    train_loader = DataLoader(train_set, args.batch_size, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_set, 1, num_workers=8)

    # define vision encoder and imitation network and combine them in the actor
    vision_encoder = make_vision_encoder()
    imi_model = Imi_networks()
    actor = robotActor(vision_encoder, imi_model)

    # define optimizer and lr scheduler
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.period, gamma=args.gamma)
    # save config
    exp_dir = save_config(args)
    # pl stuff
    pl_module = RobotLearning(actor, optimizer, train_loader, val_loader, scheduler, args)
    start_training(args, exp_dir, pl_module)

if __name__ == "__main__":
    import configargparse

    p = configargparse.ArgParser()
    p.add("-c", "--config", is_config_file=True, default="config/imi_config.yaml")
    p.add("--batch_size", default=8)
    p.add("--lr", default=0.001)
    p.add("--gamma", default=0.9)
    p.add("--period", default=3)
    p.add("--epochs", default=50)
    p.add("--resume", default=None)
    p.add("--num_workers", default=8, type=int)
    # imi_stuff
    p.add("--freeze_till", required = True, type=int)
    # data
    p.add("--train_csv", default="train.csv")
    p.add("--val_csv", default="val.csv")
    p.add("--data_folder", default="data/test_recordings")
    p.add("--resized_height", required = True, type=int)
    p.add("--resized_width", required = True, type=int)


    args = p.parse_args()
    main(args)