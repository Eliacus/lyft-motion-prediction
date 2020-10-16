import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset

from l5kit.rasterization import build_rasterizer
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from pytorch_lightning.loggers import TensorBoardLogger


class resnet_baseline(pl.LightningModule):
    """
    This is the resnet baseline model provided in
    agent_motion_prediction.ipynb, but transformed
    into a pytorch lightning module.
    """

    def __init__(self, lr, num_history_channels, cfg):
        super(resnet_baseline, self).__init__()

        self.lr = lr
        self.cfg = cfg
        # change input channels number to match the rasterizer's output
        self.num_in_channels = 3 + num_history_channels

        self.model = resnet50(pretrained=True)

        self.model.conv1 = nn.Conv2d(
            self.num_in_channels,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False,
        )

        # change output size to (X, Y) * number of future states
        num_targets = 2 * self.cfg["model_params"]["future_num_frames"]
        self.model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target_positions"]

        target_availabilities = batch["target_availabilities"].unsqueeze(-1)

        logits = self(x).reshape(y.shape)

        loss = F.mse_loss(logits, y)
        loss = loss * target_availabilities
        loss = loss.mean()

        # self.logger.experiment.add_scalar("loss", loss, global_step=batch_idx)
        self.logger.log_metrics({"loss": loss}, step=self.global_step)

        # Option 1:
        # return loss

        # Option 2:
        return {"loss": loss}

        # Option 3:
        # return {'loss': loss, 'hiddens': hiddens, 'anything_else': ...}


# TODO: Finish baselines

# TODO: create a datamodule for the lyft dataset


class LyftDataModule(pl.LightningDataModule):
    def __init__(self, data_path, config_path):
        super().__init__()
        self.data_path = data_path
        self.config_path = config_path

        os.environ["L5KIT_DATA_FOLDER"] = data_path
        self.dm = LocalDataManager(None)

        self.cfg = load_config_data(config_path)

        self.train_cfg = self.cfg["train_data_loader"]
        self.val_cfg = self.cfg["val_data_loader"]
        self.test_cfg = self.cfg["test_data_loader"]

        self.rasterizer = build_rasterizer(self.cfg, self.dm)

    def setup(self, stage=None):
        train_zarr = ChunkedDataset(self.dm.require(self.train_cfg["key"])).open()
        val_zarr = ChunkedDataset(self.dm.require(self.val_cfg["key"])).open()
        test_zarr = ChunkedDataset(self.dm.require(self.test_cfg["key"])).open()

        self.train_dataset = AgentDataset(self.cfg, train_zarr, self.rasterizer)
        self.train_dataset = AgentDataset(self.cfg, val_zarr, self.rasterizer)
        self.train_dataset = AgentDataset(self.cfg, test_zarr, self.rasterizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=self.train_cfg["shuffle"],
            batch_size=self.train_cfg["batch_size"],
            num_workers=self.train_cfg["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=self.val_cfg["shuffle"],
            batch_size=self.val_cfg["batch_size"],
            num_workers=self.val_cfg["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=self.test_cfg["shuffle"],
            batch_size=self.test_cfg["batch_size"],
            num_workers=self.test_cfg["num_workers"],
        )
