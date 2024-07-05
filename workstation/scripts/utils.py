import numpy as np
from PIL import Image

from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from sklearn.model_selection import train_test_split

"""mask processing"""
class cityscapesMaskProcessor:
    def __init__(self):
        """cityscapes color_map"""
        self.color_map =[
            (128, 64, 128), # Road
            (244, 35, 232),  # Sidewalk
            (70, 70, 70), # Building
            (102, 102, 156),  # Wall
            (190, 153, 153),  # Fence
            (153, 153, 153), # Pole
            (250, 170, 30),  # Traffic Light
            (220, 220, 0),  # Traffic Sign
            (107, 142, 35),  # Vegetation
            (152, 251, 152), # Terrain
            (70, 130, 180), # Sky
            (220, 20, 60),  # Person
            (255, 0, 0) , # Rider
            (0, 0, 142),  # Car
            (0, 0, 70) , # Truck
            (0, 60, 100) , # Bus
            (0, 80, 100) ,  # Train
            (0, 0, 230) ,   # Motorcycle
            (119, 11, 32) ,  # Bicycle
        ]


    def process_png_mask(self, mask):

        """remove alpha channel"""
        mask=mask[:,:,:3] 

        height, width = mask.shape[:2]
        num_classes = len(self.color_map)
        one_hot_mask = np.zeros((num_classes, height, width), dtype=np.uint8)
        
        for class_index,color in enumerate(self.color_map):
            binary_mask = np.all(mask == color, axis=-1)
            one_hot_mask[class_index] = binary_mask

        return one_hot_mask

class datasetSplitter:
    def __init__(self, dataset, batch_size, test_split=0.2, random_seed=5000):
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_split = test_split
        self.random_seed = random_seed
        self.train_loader = None
        self.test_loader = None

    def split(self):
        # Create a list of indices for the dataset
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))

        # Split indices into train and test
        train_indices, test_indices = train_test_split(indices, test_size=self.test_split,train_size=1-self.test_split, random_state=self.random_seed)

        # Create samplers for training and testing
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # Create data loaders
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size ,sampler=train_sampler, num_workers=4)
        self.test_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=test_sampler, num_workers=4)

        return self.train_loader, self.test_loader