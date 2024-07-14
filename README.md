# Presentation 📊

<details>
<summary>Brief Intro 🌟</summary>

- **Distillation x Segmentation**: An overview of how distillation techniques are applied to segmentation tasks.

</details>

<details>
<summary>Done ✅</summary>

<details>
<summary>Research 📚</summary>

  - **Original Paper**: Summarized insights and key points.
  - **Deep Dive into CE Loss and Dark Knowledge**: Understanding cross-entropy loss and the concept of dark knowledge in distillation.
  - **Various Paradigms of Distillation**: Exploring different distillation approaches.

</details>

<details>
<summary>Model 🧠</summary>

  - **UNet Choice Explanation**: Justification for choosing the UNet architecture for the task.

</details>

<details>
<summary>Data 📈</summary>

  - **BDD100K + Cityscapes & Class Imbalance Problem**: Analysis of the datasets used and the challenge of class imbalance.

</details>

<details>
<summary>Code 💻</summary>

  - **Working Code for Training & Distillation of UNet Model (PyTorch)**: Implementation details.
  - **Distributed Training Option (Accelerate)**: Configuration for distributed training.
  - **Logging All Chosen Model Metrics, Hardware Consumption via API (wandb)**: Monitoring and logging setup.
  - **Tested 10 Epochs on GTX 1660 TI**: Initial performance testing.

</details>

</details>

<details>
<summary>Doing 🔄</summary>

<details>
<summary>Research 🧐</summary>

  - **Thinking About a Loss Function for the Teacher Model that Favors Teaching & Performance (High Entropy Output)**: Developing a specialized loss function.
  - **Deciding on Distillation Paradigm for UNet (Decoder, Encoder, Bottleneck)**: Selecting the most suitable distillation approach for UNet.
  - **Improving Online Distillation: Alternating Loss Functions**: Exploring methods to enhance online distillation.
  - **Reframing the Problem in Terms of Bias and Variance**: Aiming for better problem formulation.

</details>

</details>

<details>
<summary>To Do 📝</summary>

<details>
<summary>Model 🛠️</summary>

  - **Aligning Number of Parameters of Student Model with Compute Capabilities of Modern Self-Driving Cars**: Optimizing the student model for real-world applications.

</details>

<details>
<summary>Data 🧬</summary>

  - **Studying Properties of the Dataset Better**: Deepening the understanding of dataset characteristics.

</details>

<details>
<summary>Code 👨‍💻</summary>

  - **Profiling Code to Measure Inference Time of the Student Model**: Performance analysis.
  - **Logging Images Through Network Layers to Gain New Insights**: Visualizing and understanding model behavior.

</details>

</details>
