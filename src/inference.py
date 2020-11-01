from pathlib import Path
from tqdm import tqdm
import numpy as np
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.evaluation import (
    compute_metrics_csv,
    create_chopped_dataset,
    read_gt_csv,
    write_pred_csv,
)
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from torch.utils.data import DataLoader
import torch
import models
from utils import convert_agent_coordinates_to_world_offsets

torch.multiprocessing.set_sharing_strategy("file_system")

# Paths
data_path = "/home/elias.nehme1/Documents/lyft-motion-prediction/data"
config_path = (
    "/home/elias.nehme1/Documents/lyft-motion-prediction/src/train_config.yaml"
)

# Load data and model
lyft_data_module = models.LyftDataModule(data_path, config_path)

model = models.resnet_baseline.load_from_checkpoint(
    "/home/elias.nehme1/Documents/lyft-motion-prediction/epoch=0-v1.ckpt"
)

# Generating a chopped dataset
num_frames_to_chop = 100
#eval_base_path = create_chopped_dataset(
#     lyft_data_module.dm.require(lyft_data_module.val_cfg["key"]),
#     lyft_data_module.cfg["raster_params"]["filter_agents_threshold"],
#     num_frames_to_chop,
#     lyft_data_module.cfg["model_params"]["future_num_frames"],
#     MIN_FUTURE_STEPS,
#)

#eval_base_path = "/home/elias.nehme1/Documents/lyft-motion-prediction/data/scenes/validate_chopped_100"
val_zarr_path = str(
    Path(eval_base_path)
    / Path(lyft_data_module.dm.require(lyft_data_module.val_cfg["key"])).name
)
val_mask_path = str(Path(eval_base_path) / "mask.npz")
val_gt_path = str(Path(eval_base_path) / "gt.csv")

val_zarr = ChunkedDataset(val_zarr_path).open()
val_mask = np.load(val_mask_path)["arr_0"]

# ===== INIT DATASET AND LOAD MASK
val_dataset = AgentDataset(
    lyft_data_module.cfg, val_zarr, lyft_data_module.rasterizer, agents_mask=val_mask
)
val_dataloader = DataLoader(
    val_dataset,
    shuffle=lyft_data_module.val_cfg["shuffle"],
    batch_size=lyft_data_module.val_cfg["batch_size"],
    num_workers=lyft_data_module.val_cfg["num_workers"],
)
print(val_dataset)

# ==== EVAL LOOP
device = torch.device("cuda")
model.to(device)
model.eval()
torch.set_grad_enabled(False)

pred_coords_list = []
confidences_list = []
timestamps_list = []
track_id_list = []

for data in tqdm(val_dataloader):
    pred, confidences = model(data["image"].to(device))
    pred = convert_agent_coordinates_to_world_offsets(
        pred.detach().cpu().numpy(),
        data["world_from_agent"].numpy(),
        data["centroid"].numpy(),
    )
    pred_coords_list.append(pred)
    confidences_list.append(confidences.detach().cpu().numpy())
    timestamps_list.append(data["timestamp"].detach().numpy())
    track_id_list.append(data["track_id"].detach().numpy())

timestamps = np.concatenate(timestamps_list)
track_ids = np.concatenate(track_id_list)
coords = np.concatenate(pred_coords_list)
confs = np.concatenate(confidences_list)

write_pred_csv(
    "predictions.csv",
    timestamps=timestamps,
    track_ids=track_ids,
    coords=coords,
    confs=confs,
)