import os
from sys import path

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import src.models as models

# Create the tensorboard logger
logger = TensorBoardLogger("tb_logs", name="my_model")

# Create the validation loss checkpoint
checkpoint_callback = ModelCheckpoint(
    filepath="tb_logs/my_model/version_17/checkpoints/",
    monitor="val_loss",
    mode="min",
)

lr = 0.003
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
    val_check_interval=100000,
    gpus=1,
    precision=16,
    limit_val_batches=100,
    resume_from_checkpoint="/home/elias/Documents/lyft-motion-prediction/tb_logs/my_model/version_17/checkpoints/epoch=0-v1.ckpt",
)

trainer.fit(model, lyft_data)
