import os
from sys import path

import torch
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import models as models

torch.multiprocessing.set_sharing_strategy("file_system")

# Create the tensorboard logger
logger = TensorBoardLogger(
    "/home/elias/Documents/lyft-motion-prediction/tb_logs/", name="multi_modal_model"
)

# Create the validation loss checkpoint
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
)

lr = 0.001
num_modes = 3

data_path = "/home/elias/Documents/lyft-motion-prediction/data"
config_path = "/home/elias/Documents/lyft-motion-prediction/src/train_config.yaml"

lyft_data = models.LyftDataModule(data_path, config_path)


model = models.resnet_baseline(lyft_data.cfg, lr, num_modes)

# model = models.resnet_baseline.load_from_checkpoint(
#    "../tb_logs/my_model/version_17/checkpoints/epoch=0-v0.ckpt-v0
# )

trainer = Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    val_check_interval=10000,
    gpus=1,
    precision=16,
    limit_val_batches=50,
)

# resume_from_checkpoint="/home/elias/Documents/lyft-motion-prediction/tb_logs/my_model/version_17/checkpoints/epoch=1.ckpt",

trainer.fit(model, lyft_data)
