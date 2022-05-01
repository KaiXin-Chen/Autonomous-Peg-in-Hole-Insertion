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
            # ADE and maximum_diff can be used as metrics for a batch as well.
            #ADE = np.average(np.sqrt(np.sum(np.square(pred[:,0:3]-demo[:,0:3]),axis=1)),axis=0)
            #maximum_diff = np.max(np.sqrt(np.sum(np.square(pred[:,0:3]-demo[:,0:3]),axis=1)))
            # FDE only make sense when the demo/pred is sequential
            #FDE = np.sqrt(np.sum(np.square(pred[-1:,0:3]-demo[-1:,0:3]),axis=1))
            # pred is (batch_size, 3, 3)
            # demo is (batch_size, 3)
            return self.cce(pred, demo)
        # TODO: Implement the below training steps, how to calculate loss and accuracy
        vision_img, gt_action = batch
        vision_img = Variable(vision_img).cuda()
        gt_action = Variable(gt_action).cuda()
        logits = self.actor(vision_img, self.current_epoch < self.config.freeze_till)
        # The main task here is to reshape and normalize logits and ground truth such that we can apply cross entropy on the results as well as being able to calculate the accuracy
        N,_ =logits.size()
        # reshape to 3 channels per sample, which is x,y,z, and each channel has three classes negitive 0 positive so nx3x3
        logits=logits.view(N,3,3)
        # convert them to probabilities
        # logits= torch.nn.functional.softmax(logits, -1)
        # convert gt action to (N,3,3), where at axis=-1 it is a one hot vector
        # gt_action=torch.sign(gt_action)

        # gt_act_1hot= torch.zeros(N,3,3)
        # gt_act_1hot[:, :, 0]=1*(gt_action<-0.5)
        # gt_act_1hot[:, :, 2] = 1 * ( gt_action > 0.5)
        # gt_act_1hot[:, :, 1] = 1 - gt_act_1hot[:, :, 0] - gt_act_1hot[:, :, 2]
        # gt_act_1hot = gt_act_1hot.cuda(0)
        loss = compute_loss(logits, gt_action)
        train_acc = torch.sum(torch.argmax(logits,axis=-1)==gt_action)/N/3

        values = {'train/loss': loss, 'train/acc': 0}
        #self.log_dict(values)
        return {'loss':loss,'accu':train_acc}



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

        vision_img, gt_action = batch
        vision_img = Variable(vision_img).cuda()
        gt_action = Variable(gt_action).cuda()
        logits = self.actor(vision_img, self.current_epoch < self.config.freeze_till)
        N, _ = logits.size()
        logits = logits.view(N, 3, 3)
        # logits = torch.nn.functional.softmax(logits, -1)
        # gt_action=torch.sign(gt_action)
        # gt_act_1hot = torch.zeros(N, 3, 3)
        # gt_act_1hot[:, :, 0] = 1 * (gt_action < 0.5)
        # gt_act_1hot[:, :, 2] = 1 * (gt_action > 1.5)
        # gt_act_1hot[:, :, 1] = 1- gt_act_1hot[:, :, 0]-gt_act_1hot[:, :, 2]

        # gt_act_1hot = gt_action.cuda(0)
        loss = compute_loss(logits, gt_action)
        val_acc = torch.sum(torch.argmax(logits,axis=-1)==gt_action)/N/3
        values = {'val/loss': loss, 'val/acc': 0}
        #self.log_dict(values)
        return {'loss':loss,'accu':val_acc}


    def validation_epoch_end(self, val_step_outputs) :
        val_acc=torch.tensor([ dict['accu'] for dict in val_step_outputs]).cuda()
        values = {'val/epoch_acc': torch.mean(val_acc)}
        self.log_dict(values)


    def train_epoch_end(self, train_step_outputs) :
        train_acc=torch.tensor([ dict['accu'] for dict in train_step_outputs]).cuda()
        values = {'train/epoch_acc': torch.mean(train_acc)}
        self.log_dict(values)

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_set_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_set_loader

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]