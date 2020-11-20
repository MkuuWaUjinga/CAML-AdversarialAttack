import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from main import test, Net

epsilons = [0, .05, .1, .15, .2, .25, .3 ]
pretrained_model = "data/lenet_mnist_model.pth"
# Only do attack on first k test data to speed up run time
first_k = 10000
use_cuda = False
runs = 5

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

# Run test for each FGSM-iteration
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps, runs, first_k)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.title("Accuracy vs Epsilon, iterations = 10")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        # Limit width to 5 cols
        plt.subplot(len(examples[0])*len(epsilons)//5 + 1, 5, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex, class_dist, perturbed_hist, noise_hist = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
