from l5kit.evaluation import (
    write_pred_csv,
    compute_metrics_csv,
    read_gt_csv,
    create_chopped_dataset,
)
import models
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS

# Paths
data_path = "/home/elias/Documents/lyft-motion-prediction/data"
config_path = "/home/elias/Documents/lyft-motion-prediction/src/train_config.yaml"

# Load data and model
lyft_data_module = models.LyftDataModule(data_path, config_path)

model = models.resnet_baseline.load_from_checkpoint(
    "/home/elias/Documents/lyft-motion-prediction/tb_logs/my_model/version_17/checkpoints/epoch=1.ckpt"
)

# Generating a chopped dataset
num_frames_to_chop = 100
eval_base_path = create_chopped_dataset(
    lyft_data_module.dm.require(lyft_data_module.val_cfg["key"]),
    lyft_data_module.cfg["raster_params"]["filter_agents_threshold"],
    num_frames_to_chop,
    lyft_data_module.cfg["model_params"]["future_num_frames"],
    MIN_FUTURE_STEPS,
)
