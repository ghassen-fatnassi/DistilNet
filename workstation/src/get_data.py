from config.config_loader import load_yaml
import dataset_tools as dtools

config=load_yaml()

key=input("do u want to download the CARLA simulator dataset? press Y for yes , press any other key for no")
if key=="Y" or key=="y":
    dtools.download(dataset='Self Driving Cars', dst_dir=config['paths']['datasets'])
print("Done!")
