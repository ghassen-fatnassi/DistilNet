import yaml
import json
import dataset_tools as dtools

def load_yaml(config_path="/media/gaston/gaston1/DEV/ACTIA/workstation/config/config.yaml"):
    with open(config_path,'r') as file:
        config=yaml.safe_load(file)
    return config