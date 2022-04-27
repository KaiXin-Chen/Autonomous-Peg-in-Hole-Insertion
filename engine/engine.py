"""
This file defines the training and validation steps
"""
import time
import cv2
from pytorch_lightning import LightningModule
from tomlkit import key
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision

class RobotLearning(LightningModule):
    def __init__(self, actor, optimizer, train_set_loader, val_set_loader, scheduler, config):
        """
        :param actor:
            The models you write including vision encoders and imitation learning
        :param optimizer:
            The optimizer you are using, default to Adam
        :param train_set_loader:
            The training dataset loader you defined
        :param val_set_loader:
            The validation dataset you defined
        :param scheduler:
            The learning rate scheduler you defined
        :param config:
            The config file you use for training
        """
        super().__init__()
        self.actor = actor
        self.optimizer = optimizer
        self.train_set_loader = train_set_loader
        self.val_set_loader = val_set_loader
        self.scheduler = scheduler
        self.config = config

    def training_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred:
                The predictions from the model
            demo:
                The ground truth action from the human demonstration
            """
            raise NotImplementedError
        # TODO: Implement the below training steps, how to calculate loss and accuracy
        vision_img, gt_action = batch
        pred_action = self.actor(vision_img, self.current_epoch < self.config.freeze_until)
        loss = compute_loss(pred_action, gt_action)
        train_acc = 0
        self.log_dict({"train/loss": loss})
        self.log_dict({"train/acc": train_acc})
        return loss

    def validation_step(self, batch, batch_idx):
        
        def compute_loss(pred, demo):
            """
            pred:
                The predictions from the model
            demo:
                The ground truth action from the human demonstration
            """
            raise NotImplementedError
        # TODO: Implement the below validation steps, how to calculate loss and accuracy
        vision_img, gt_action = batch
        with torch.no_grad():
            pred_action = self.actor(vision_img, True)
            loss = compute_loss(pred_action, gt_action)
            val_acc =0
        self.log_dict({"val/loss": loss})
        self.log_dict({"val/acc": val_acc})
        return loss

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]