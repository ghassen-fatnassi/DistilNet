# Autonomous Driving Segmentation Task 🚗🎨

This project focuses on segmentation tasks for autonomous driving using the Cityscapes dataset. It supports distributed training and mixed precision and includes a logging mechanism with WandB to log important metrics.

## Folder Structure 📂

```
workstation
├── 📁 data
│   └── 📁 cityscapes
│       ├── 📁 images 🖼️ (contains the dataset images)
│       └── 📁 masks 🎭 (contains the dataset masks)
├── 📁 logs 📝
│   └── 📁 wandb 📊 (contains the WandB logs)
├── 📁 models 🤖
│   ├── 📁 Students 👩‍🎓 (contains the student models in pth format)
│   └── 📁 Teachers 👨‍🏫 (contains the teacher models in pth format)
├── 📁 src 📦
│   ├── 📁 config 🛠️
│   │   ├── 📄 config.yaml 📝 (contains the project configuration)
│   │   ├── 📄 SegFormer.yaml 📝 (contains the SegFormer model configuration)
│   │   └── 📄 UNet.yaml 📝 (contains the UNet model configuration)
│   ├── 📁 models 🤖
│   │   └── 📄 UNet.py 📝 (contains the UNet model implementation)
│   └── 📁 training 🏋️‍♂️
│       ├── 📄 train.py 📝 (contains the training script)
│       ├── 📄 engine.py 📝 (contains the training engine)
│       ├── 📄 dataset.py 📝 (contains the dataset class)
│       ├── 📄 loss.py 📝 (contains the custom loss functions)
│       ├── 📄 distill.py 📝 (contains the distillation logic)
│       ├── 📄 utils.py 📝 (contains utility functions)
│       └── 📄 init.py 📝 (contains initialization code)
├── 📄 .gitignore 📝
├── 📄 LICENSE 📝
├── 📄 README.md 📝
├── 📄 requirements.txt 📝
└── 📄 x.ipynb 📝
```

## Done ✅

- Downloaded and structured the Cityscapes dataset (images and masks).
- Created a `Dataset` class to handle dataset intricacies for future customization if I ever introduce new datasets.
- Developed a `DataSplitter` class to create train and validation dataloaders with random sampling.
- Made the project modular with custom loss functions, custom models, and configuration files.
- Implemented logging with WandB to track metrics like loss, IOU, F1, recall, and mAP.
- Supports distributed training.

## To Do 📝

- Log images through epochs, layers, and between models to gain insights into model performance on images.

## Doing 🛠️

- Ongoing improvements and bug fixes.
- Adding additional logging and visualization features.
- Supporting mixed precision through configuration files.

## Configuration and Dependencies 🛠️

- **Frameworks:** PyTorch, Accelerate
- **Logging:** WandB

## Example Image 🖼️

Here is an example of an image similar to what the project handles in segmentation tasks:

![Segmentation Example](https://www.cityscapes-dataset.com/example_image.png)

## Usage 🚀

1. **Clone the repository:**

```bash
git clone <repository_url>
cd <repository_name>
```

2. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

3. **Prepare the dataset:**
   Place the Cityscapes dataset images and masks in the `data/cityscapes` folder.

4. **Run training:**

```bash
python src/training/train.py --config src/config/config.yaml
```

## Contributing 🤝

Feel free to open issues or submit pull requests with improvements.

## License 📄

This project is licensed under the MIT License.
