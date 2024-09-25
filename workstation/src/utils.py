import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset,random_split
import yaml
import json

torch.manual_seed(50)

"""loading config file"""
def load_yaml(config_path="./actia/workstation/src/config/config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

"""overwriting they yaml file"""
def change_yaml(new_dict,config_path="./actia/workstation/src/config/config.yaml"):
    with open(config_path, 'w') as file:
        yaml.dump(new_dict,file,default_flow_style=False)

class UnetMaskProcessor:
    def __init__(self):
        """Cityscapes color_map"""
        self.color_map = [
            (128, 64, 128),   # Road
            (244, 35, 232),   # Sidewalk
            (70, 70, 70),     # Building
            (102, 102, 156),  # Wall
            (190, 153, 153),  # Fence
            (153, 153, 153),  # Pole
            (250, 170, 30),   # Traffic Light
            (220, 220, 0),    # Traffic Sign
            (107, 142, 35),   # Vegetation
            (152, 251, 152),  # Terrain
            (70, 130, 180),   # Sky
            (220, 20, 60),    # Person
            (255, 0, 0),      # Rider
            (0, 0, 142),      # Car
            (0, 0, 70),       # Truck
            (0, 60, 100),     # Bus
            (0, 80, 100),     # Train
            (0, 0, 230),      # Motorcycle
            (119, 11, 32),    # Bicycle
        ]
        self.class_labels = {
            0: 'Road', 
            1: 'Sidewalk', 
            2: 'Building', 
            3: 'Wall', 
            4: 'Fence', 
            5: 'Pole', 
            6: 'Traffic Light', 
            7: 'Traffic Sign', 
            8: 'Vegetation', 
            9: 'Terrain', 
            10: 'Sky', 
            11: 'Person', 
            12: 'Rider', 
            13: 'Car', 
            14: 'Truck',
            15: 'Bus', 
            16: 'Train', 
            17: 'Motorcycle', 
            18: 'Bicycle'
        }

        self.num_classes = len(self.color_map)
        
    def one_hotting_mask(self, mask):
        """Convert RGB mask to one-hot encoded mask"""
        height, width,_ = mask.shape
        one_hot_mask = np.zeros((self.num_classes, height, width), dtype=np.uint8)

        # Iterate over each pixel to find class index
        for class_index, color in enumerate(self.color_map):
            binary_mask = np.all(mask == color, axis=-1)
            one_hot_mask[class_index] = binary_mask
        return one_hot_mask

class datasetSplitter:
    def __init__(self, dataset, batch_size, test_split=0.1, random_seed=7):
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_split = test_split
        self.random_seed = random_seed
        self.train_loader = None
        self.test_loader = None

    def split(self):
        # Create a list of indices for the dataset
        dataset_size = len(self.dataset)

        train_dataset, test_dataset = random_split(self.dataset, [dataset_size-int(self.test_split * dataset_size),int(self.test_split * dataset_size)])
        
        train_dataset.dataset.augment=True
        test_dataset.dataset.augment=False

        # Create data loaders
        self.train_loader = DataLoader(train_dataset,batch_size=self.batch_size ,num_workers=4,persistent_workers=True,shuffle=True,prefetch_factor=64)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=2,shuffle=False,prefetch_factor=32)

        return self.train_loader, self.test_loader
