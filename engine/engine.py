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
        self.cce = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        def compute_loss(pred, demo):
            """
            pred:
                The predictions from the model
            demo:
                The ground truth action from the human demonstration
            """
            # pred is (batch_size, 3, 3)
            # demo is (batch_size, 3)
            return self.cce(pred, demo)
        # TODO: Implement the below training steps, how to calculate loss and accuracy
        vision_img, gt_action,pos = batch
        vision_img = Variable(vision_img.type(torch.FloatTensor)).cuda()
        gt_action = Variable(gt_action.type(torch.LongTensor)).cuda()
        pos = Variable(pos.type(torch.FloatTensor)).cuda()
        logits = self.actor(vision_img,pos, self.current_epoch < self.config.freeze_till)
        # The main task here is to reshape and normalize logits and ground truth such that we can apply cross entropy on the results as well as being able to calculate the accuracy
        N, _ =logits.size()
        # reshape to 3 channels per sample, which is x,y,z, and each channel has three classes negitive 0 positive so nx3x3
        logits=logits.view(N,3,3)
        loss = compute_loss(logits, gt_action)
        action_pred = torch.argmax(logits, dim=-1)
        values = {'train/loss': loss}
        self.log_dict(values)
        return  {"loss":loss,"correct_count": (action_pred == gt_action).sum(), "total_count":gt_action.numel() }

    def validation_step(self, batch, batch_idx):
        
        def compute_loss(pred, demo):
            """
            pred:
                The predictions from the model
            demo:
                The ground truth action from the human demonstration
            """
            return torch.nn.functional.cross_entropy(pred, demo)
        # TODO: Implement the below validation steps, how to calculate loss and accuracy

        vision_img, gt_action,pos = batch
        vision_img = Variable(vision_img.type(torch.FloatTensor)).cuda()
        gt_action = Variable(gt_action.type(torch.LongTensor)).cuda()
        pos=Variable(pos.type(torch.FloatTensor)).cuda()
        logits = self.actor(vision_img, pos,self.current_epoch < self.config.freeze_till)
        N, _ = logits.size()
        logits = logits.view(N, 3, 3)
        loss = compute_loss(logits, gt_action)
        action_pred = torch.argmax(logits, dim=-1)
        values = {'val/loss': loss}
        self.log_dict(values)
        return loss, ((action_pred == gt_action).sum(), gt_action.numel())

    def validation_epoch_end(self, val_step_outputs):
        num = 0
        de = 0

        for val_step_output in val_step_outputs:
            loss, stat = val_step_output
            corr, total= stat
            num += corr
            de += total

        acc = num / de
        values = {'val/acc': acc}
        self.log_dict(values)#, on_step=False, on_epoch=True)

    def train_epoch_end(self, train_step_outputs):
        num = 0
        de = 0
        for train_step_output in train_step_outputs:
            loss=train_step_output["loss"]
            corr=train_step_output["correct_count"]
            total = train_step_output["total_count"]
            num += corr
            de += total
        acc = num / de
        values = {'train/epoch_acc': acc}
        self.log_dict(values, on_step=False, on_epoch=True)


    def train_dataloader(self):
        """Training dataloader"""
        return self.train_set_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_set_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]