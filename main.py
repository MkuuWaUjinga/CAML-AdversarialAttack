import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted code from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html to
# support iterative fast gradient sign method (i-FGSM) attacks as well.
# In case you want to perform FGSM just set runs = 1.

# TODO do step wise example creation
# Num columns = num iterations
# First Row = Current Perturbed Image
# Second Row = Current Added Noise
# Third Row = Current Class Distribution


def fgsm_attack(image, epsilon, data_grad):
    # Get sign of gradients for each input pixel
    gradient_sign = data_grad.sign()
    # Perturb image by doing gradient ascent with respect to the pixels
    perturbed_image = image + epsilon * gradient_sign
    # Make sure that input values of image stay the same.
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, epsilon * gradient_sign

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def test(model, device, test_loader, epsilon, num_runs, first_k):

    # Accuracy counter
    num_samples_seen = 0
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        noise = torch.zeros_like(data).to(device)
        class_dist = []
        perturbed_hist = [data.squeeze().detach().cpu().numpy()]
        noise_hist = [noise.squeeze().detach().cpu().numpy()]


        # Send the data and label to the device
        data, target = data.to(device), target.to(device)


        assert num_runs > 0, "Please do at least one round of FGSM attack"
        perturbed_data = data

        # Set requires_grad attribute of tensor. Important for Attack
        perturbed_data.requires_grad = True

        # Do iterative FGSM attack
        try:
            for run in range(num_runs):

                # Forward pass the data through the model
                output = model(perturbed_data)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

                # If the initial prediction is wrong, dont bother attacking, just move on
                if pred.item() != target.item() and run == 0:
                    raise ValueError("Initial Prediction Wrong")
                # If the label has been flipped after an attack, stop doing iterations.
                elif pred.item() != target.item() and run != 0:
                    print(f"Label flipped after {run} iterations")
                    #break

                # Calculate the loss
                loss = F.nll_loss(output, target)

                # Zero all existing gradients
                model.zero_grad()

                # Calculate gradients of model in backward pass
                loss.backward()

                # Collect datagrad
                data_grad = perturbed_data.grad.data

                # Do iterative FGSM Attack
                perturbed_data, noise = fgsm_attack(perturbed_data, epsilon/num_runs, data_grad)
                perturbed_data = perturbed_data.detach()
                perturbed_data.requires_grad = True

                # Append current class distribution
                class_dist.append(output.squeeze().detach().cpu().numpy())
                perturbed_hist.append(perturbed_data.squeeze().detach().cpu().numpy())
                noise_hist.append(noise_hist[run - 1] + noise.squeeze().detach().cpu().numpy())

        except ValueError:
            continue

        # Re-classify the perturbed image
        output = model(perturbed_data)
        class_dist.append(output.squeeze().detach().cpu().numpy())

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                # We can use the target here because we made sure that target == init_prediction
                adv_examples.append( (target.item(), final_pred.item(), adv_ex, class_dist, perturbed_hist, noise_hist) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (target.item(), final_pred.item(), adv_ex, class_dist, perturbed_hist, noise_hist) )

        num_samples_seen += 1
        # Enough attacks for today?
        if num_samples_seen == first_k:
            break

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(num_samples_seen)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, num_samples_seen, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

