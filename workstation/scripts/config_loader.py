import yaml
import json
import dataset_tools as dtools

def load_yaml(config_path="FOR-ACTIA/FF/config/config.yaml"):
    with open(config_path,'r') as file:
        config=yaml.safe_load(file)
    return config

def load_json(json_path="FOR-ACTIA/FF/config/hyperparameters.json"):
    with open(json_path,'r') as file:
        config = json.load(file)
    return config