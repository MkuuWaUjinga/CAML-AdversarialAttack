import torch
import matplotlib.pyplot as plt
from scipy.special import softmax
from torchvision import datasets, transforms
from main import test, Net
import numpy as np

epsilon = .1
pretrained_model = "data/lenet_mnist_model.pth"
# Only do attack on first k test data to speed up run time
first_k = 100
use_cuda = False
runs = 10

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=1, shuffle=False)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

accuracies = []
examples = []

acc, ex = test(model, device, test_loader, epsilon, runs, first_k)
accuracies.append(acc)
examples.append(ex)

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(12,8))
indices = [0]
for j in indices:#range(len(examples[0])):
    orig, adv, ex, class_dist, perturbed_hist, noise_hist = examples[0][j]
    for step, triple in enumerate(zip(class_dist, perturbed_hist, noise_hist)):
        if step % 2 == 1:
            continue
        dis, per, noise = triple
        cnt += 1
        # Plot image
        plt.subplot(4, 6, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(f"Iteration {step}")
        plt.imshow(per, cmap="gray")
        if step == 0:
            plt.ylabel("Attack Image", fontsize=14)
        # Plot noise. The grayer the pixel the stronger the change.
        plt.subplot(4, 6, cnt + 6)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(1 - np.abs(noise), cmap="gray", vmin=0.5, vmax=1)
        if step == 0:
            plt.ylabel("Noise", fontsize=14)
        # Plot claas_dist
        plt.subplot(4, 6, cnt + 12)
        plt.bar(np.arange(len(dis)), softmax(dis))
        plt.yticks(np.arange(0, 1, step=0.2))
        plt.xlabel("Class")
        plt.ylabel("Probability")

plt.tight_layout()
plt.show()