import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50

from utils import pytorch_neg_multi_log_likelihood_batch


class resnet_baseline(pl.LightningModule):
    """
    This is the resnet baseline model provided in
    agent_motion_prediction.ipynb, but transformed
    into a pytorch lightning module.
    """

    def __init__(self, cfg, lr, num_modes):
        super(resnet_baseline, self).__init__()
        self.save_hyperparameters()

        self.cfg = self.hparams.cfg  # type: ignore
        # self.lr = self.cfg["train_params"]["learning_rate"]  # type: ignore
        # self.num_modes = self.cfg["train_params"]["num_modes"]  # type: ignore
        self.lr = lr
        self.num_modes = num_modes
        # change input channels number to match the rasterizer's output
        num_history_channels = (self.cfg["model_params"]["history_num_frames"] + 1) * 2
        self.num_in_channels = 3 + num_history_channels
        self.future_len = cfg["model_params"]["future_num_frames"]

        self.model = resnet50(pretrained=True)

        self.model.conv1 = nn.Conv2d(
            self.num_in_channels,
            self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False,
        )

        # change output size to (X, Y) * number of future states * num modes
        num_targets = 2 * self.cfg["model_params"]["future_num_frames"]
        self.num_preds = num_targets * self.num_modes

        self.model.fc = nn.Sequential(
            nn.Linear(
                in_features=2048,
                # num of modes * preds + confidence
                out_features=self.num_preds + self.num_modes,
            ),
        )

        # Loss function
        self.criterion = pytorch_neg_multi_log_likelihood_batch

    def forward(self, data):
        out = self.model(data)
        batch_size = data.shape[0]
        pred, confidences = torch.split(out, self.num_preds, dim=1)
        assert pred.shape == (batch_size, self.num_preds)
        assert confidences.shape == (batch_size, self.num_modes)
        pred = pred.view(batch_size, self.num_modes, self.future_len, 2)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target_positions"]

        target_availabilities = batch["target_availabilities"]

        pred, confidences = self(x)

        loss = self.criterion(y, pred, confidences, target_availabilities)

        self.logger.log_metrics(
            {
                "train_loss": loss,
                "confidence_sum": torch.sum(confidences, dim=1)[0],
                "confidences_1": confidences[0][0],
                "confidences_2": confidences[0][1],
                "confidences_3": confidences[0][2],
            },
            step=self.global_step,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target_positions"]

        target_availabilities = batch["target_availabilities"]

        pred, confidences = self(x)

        loss = self.criterion(y, pred, confidences, target_availabilities)

        self.logger.log_metrics({"val_loss": loss}, step=self.global_step)

        self.log("val_loss", loss)

        return loss


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
        self.train_dataset = AgentDataset(self.cfg, train_zarr, self.rasterizer)

        val_zarr = ChunkedDataset(self.dm.require(self.val_cfg["key"])).open()
        self.val_dataset = AgentDataset(self.cfg, val_zarr, self.rasterizer)

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
        test_zarr = ChunkedDataset(self.dm.require(self.val_cfg["key"])).open()
        self.test_dataset = AgentDataset(
            self.cfg, test_zarr, self.rasterizer, agents_mask=test_mask
        )

        return DataLoader(
            self.test_dataset,
            shuffle=self.test_cfg["shuffle"],
            batch_size=self.test_cfg["batch_size"],
            num_workers=self.test_cfg["num_workers"],
        )
