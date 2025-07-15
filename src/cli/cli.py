import yaml
from train_entry import TrainConfig, train_model_from_config

with open("configs/base.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

cfg = TrainConfig(**config_dict)
model = train_model_from_config(cfg)
