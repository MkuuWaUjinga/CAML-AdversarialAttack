# CAML-AdversarialAttack
This repository is part of a seminar work on automatic differentiation and its importance for machine learning. 
The repository implements the FGSM and i-FGSM attack methods. 

The code was adapted from [this pytorch tutorial](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html). Please dowload the trained model via [this link](https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h) and place it in your data folder. 

Here is an image of a successful i-FGSM attack with 10 iterations. A 4 becomes a 8. Do you see the difference?

![Alt](./images/shift4->8_cropped.png?raw=true, "Successful i-FGSM Attack on MNIST")
