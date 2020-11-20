import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from main import test, Net

epsilons = [.1]
pretrained_model = "data/lenet_mnist_model.pth"
# Only do attack on first k test data to speed up run time
first_k = 1000
use_cuda = False
runs = [1, 3, 5, 10, 20]

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

accuracies = [0.981]

# Run test for each FGSM-iteration
for eps in epsilons:
    for run in runs:
        acc, ex = test(model, device, test_loader, eps, run, first_k)
        accuracies.append(acc)

runs = [0] + runs

plt.figure(figsize=(5,5))
plt.plot(runs, accuracies, "*-")
plt.title("Accuracy vs FGSM iterations, epsilon = 0.1")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.show()
