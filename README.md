# Project Overview

## Introduction
**Distillation in Semantic Segmentation**: An exploration of how knowledge distillation techniques can enhance model efficiency and accuracy in segmentation tasks.

## Completed Tasks

### Research
- **Paper Analysis**: Summarized insights from the original paper, focusing on key methodologies and findings.
- **Cross-Entropy Loss and Dark Knowledge**: Reviewed the role of cross-entropy loss and the concept of "dark knowledge" in knowledge distillation.
- **Distillation Paradigms**: Explored various distillation methods, assessing their applicability to segmentation.

### Model Development
- **UNet Architecture Selection**: Provided justification for using the UNet architecture for this segmentation task, considering its compatibility with distillation techniques.

### Data Preparation
- **Dataset Analysis (BDD100K and Cityscapes)**: Evaluated dataset suitability for segmentation tasks and addressed class imbalance issues.

### Code Implementation
- **UNet Training and Distillation Pipeline (PyTorch)**: Built and tested a comprehensive codebase for training and distillation.
- **Distributed Training Configuration (Accelerate)**: Configured for efficient distributed training using the Accelerate library.
- **Logging and Monitoring (Weights & Biases)**: Set up monitoring to track model metrics, hardware usage, and other performance indicators.
- **Initial Model Testing**: Conducted preliminary testing over 10 epochs on a GTX 1660 TI to verify setup and performance.

## Ongoing Work

### Research
- **Loss Function for Teacher Model**: Designing a loss function that promotes high entropy outputs, supporting effective knowledge transfer.
- **Distillation Paradigm Selection**: Identifying the best distillation approach for UNet, considering encoder, decoder, or bottleneck-focused strategies.
- **Improving Online Distillation**: Experimenting with alternating loss functions to enhance online distillation.
- **Reframing the Problem with Bias and Variance**: Formulating the problem with a focus on bias-variance trade-offs to improve model generalization.

## Next Steps

### Model Optimization
- **Parameter Reduction for Student Model**: Adapting the student model's parameters to align with the computational capabilities of autonomous driving systems.

### Data Exploration
- **In-Depth Dataset Analysis**: Further study of dataset characteristics to better understand their impact on model performance.

### Code and Performance Analysis
- **Inference Time Profiling**: Profiling and optimizing the inference time of the student model to ensure real-time performance.
- **Layer-wise Image Logging**: Developing visualization tools to track data flow through network layers, providing insights into model behavior.
