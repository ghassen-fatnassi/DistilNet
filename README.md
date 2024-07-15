# Presentation 📊

## Brief Intro 🌟
- **Distillation x Segmentation**: An overview of how distillation techniques are applied to segmentation tasks.

## Done ✅
- **Research 📚**
  - **Original Paper**: Summarized insights and key points.
  - **Deep Dive into CE Loss and Dark Knowledge**: Understanding cross-entropy loss and the concept of dark knowledge in distillation.
  - **Various Paradigms of Distillation**: Exploring different distillation approaches.
- **Model 🧠**
  - **UNet Choice Explanation**: Justification for choosing the UNet architecture for the task.
- **Data 📈**
  - **BDD100K + Cityscapes & Class Imbalance Problem**: Analysis of the datasets used and the challenge of class imbalance.
- **Code 💻**
  - **Working Code for Training & Distillation of UNet Model (PyTorch)**: Implementation details.
  - **Distributed Training Option (Accelerate)**: Configuration for distributed training.
  - **Logging All Chosen Model Metrics, Hardware Consumption via API (wandb)**: Monitoring and logging setup.
  - **Tested 10 Epochs on GTX 1660 TI**: Initial performance testing.

## Doing 🔄
- **Research 🧐**
  - **Thinking About a Loss Function for the Teacher Model that Favors Teaching & Performance (High Entropy Output)**: Developing a specialized loss function.
  - **Deciding on Distillation Paradigm for UNet (Decoder, Encoder, Bottleneck)**: Selecting the most suitable distillation approach for UNet.
  - **Improving Online Distillation: Alternating Loss Functions**: Exploring methods to enhance online distillation.
  - **Reframing the Problem in Terms of Bias and Variance**: Aiming for better problem formulation.

## To Do 📝
- **Model 🛠️**
  - **Aligning Number of Parameters of Student Model with Compute Capabilities of Modern Self-Driving Cars**: Optimizing the student model for real-world applications.
- **Data 🧬**
  - **Studying Properties of the Dataset Better**: Deepening the understanding of dataset characteristics.
- **Code 👨‍💻**
  - **Profiling Code to Measure Inference Time of the Student Model**: Performance analysis.
  - **Logging Images Through Network Layers to Gain New Insights**: Visualizing and understanding model behavior.
