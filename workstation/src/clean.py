import torch
import math
import matplotlib.pyplot as plt

# Define initial settings
initial_lr = 0.05
T_0 = 5
T_mult = 2
eta_min = 0.0005
epochs = 100

# Define the optimizer with a dummy model parameter
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

# Define the schedulers
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[3])

# Store the learning rates for each epoch
learning_rates = []

for epoch in range(epochs):
    optimizer.step()
    learning_rates.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# Plot the learning rate progression
plt.plot(range(1, epochs+1), learning_rates, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Progression')
plt.grid(True)
plt.show()
