import os
from pathlib import Path

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.evaluation import (
    compute_metrics_csv,
    create_chopped_dataset,
    read_gt_csv,
    write_pred_csv,
)
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import (
    PREDICTED_POINTS_COLOR,
    TARGET_POINTS_COLOR,
    draw_trajectory,
)
from prettytable import PrettyTable
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

import models

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "data"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("src/train_config.yaml")


lr = 0.003
num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

data_path = "/home/elias/Documents/lyft-motion-prediction/data"
config_path = "/home/elias/Documents/lyft-motion-prediction/src/train_config.yaml"

lyft_data = models.LyftDataModule(data_path, config_path)

model = models.resnet_baseline(lr, num_history_channels, lyft_data.cfg)

trainer = Trainer(gpus=1, precision=16)

trainer.fit(model, lyft_data)
