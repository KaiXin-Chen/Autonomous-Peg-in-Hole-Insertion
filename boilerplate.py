"""
This files includes the boilerplate of all training scripts
"""

import os
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import numpy as np


def save_config(args):
    """
    :param args:
        includes all arguments used in training will store them during trianing
    :return:
    """
    config_name = os.path.basename(args.config).split(".yaml")[0]
    now = datetime.now()
    dt = now.strftime("%m%d%Y")
    exp_dir = os.path.join("exp" + dt, config_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "conf.yaml"), "w") as outfile:
        yaml.safe_dump(vars(args), outfile)
    return exp_dir


def start_training(args, exp_dir, pl_module, monitor="val/acc"):
    """
    :param args:
        include all training config arguments
    :param exp_dir:
        include the diretory for the experiment results
    :param pl_module:
        include the models
    :param monitor:
        include the metric you are tracking during training

    This function just put everything we write to the framework of pytorch lightning
    """
    # This part defines how you save the checkpoints during training
    exp_time = datetime.now().strftime("%m-%d-%H:%M:%S") + "-v" + str(np.random.randint(0, 1000))
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, "checkpoints"),
        filename=exp_time + "{epoch}-{step}",
        save_top_k=1,
        save_last=True,
        monitor=monitor,
        mode='max'
    )

    # This part defines how you save the logger file for tensorboard use
    logger = TensorBoardLogger(save_dir=exp_dir, version=exp_time, name="lightning_logs")
    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint],
        default_root_dir=exp_dir,
        gpus=-1,
        strategy="dp",
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        logger=logger
    )

    # This part defines how the trainer fit the model and resume training
    trainer.fit(
        pl_module,
        ckpt_path=None
        if args.resume is None
        else os.path.join(os.getcwd(), args.resume),
    )
    print("best_model", checkpoint.best_model_path)
