import os

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import models

logger = TensorBoardLogger("tb_logs", name="my_model")

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "data"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("src/train_config.yaml")


lr = 0.0003
num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

# data_path = "/home/elias/Documents/lyft-motion-prediction/data"
# config_path = "/home/elias/Documents/lyft-motion-prediction/src/train_config.yaml"

data_path = "/home/elias.nehme1/Documents/lyft-motion-prediction/data"
config_path = (
    "/home/elias.nehme1/Documents/lyft-motion-prediction/src/train_config.yaml"
)

lyft_data = models.LyftDataModule(data_path, config_path)

model = models.resnet_baseline(lr, num_history_channels, lyft_data.cfg)

trainer = Trainer(gpus=1, logger=logger)

trainer.fit(model, lyft_data)
