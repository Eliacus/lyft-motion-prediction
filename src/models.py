import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
import torch.functional as F


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
        return torch.optim.Adam(self.parameters, lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x).reshape(y.shape)

        loss = F.mse_loss(logits, y)

        result = pl.TrainResult(loss)

        result.log("train_loss", loss, prog_bar=True)

        return result


# TODO: Finish baselines

# TODO: create a datamodule for the lyft dataset
