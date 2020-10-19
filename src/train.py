import os

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import models


# Create the tensorboard logger
logger = TensorBoardLogger("tb_logs", name="my_model")

# Create the validation loss checkpoint
checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

lr = 0.003
num_modes = 3

data_path = "/home/elias/Documents/lyft-motion-prediction/data"
config_path = "/home/elias/Documents/lyft-motion-prediction/src/train_config.yaml"

# data_path = "/home/elias.nehme1/Documents/lyft-motion-prediction/data"
# config_path = (
#    "/home/elias.nehme1/Documents/lyft-motion-prediction/src/train_config.yaml"
# )

lyft_data = models.LyftDataModule(data_path, config_path)

model = models.resnet_baseline(lyft_data.cfg, lr, num_modes)

trainer = Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    val_check_interval=0.1,
    gpus=1,
    precision=16,
)

trainer.fit(model, lyft_data)
