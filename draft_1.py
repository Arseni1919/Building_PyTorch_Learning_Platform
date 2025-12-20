import torch
import torchvision
import torchvision.transforms as transforms
import os

# Define your transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Check if the dataset exists
data_path = './data'
cifar_path = os.path.join(data_path, 'cifar-10-batches-py')

if os.path.exists(cifar_path):
    print("CIFAR-10 dataset found!")
    print("Contents:", os.listdir(cifar_path))
else:
    print("CIFAR-10 dataset not found. Please check the extraction.")
    print("Looking for:", cifar_path)
    if os.path.exists(data_path):
        print("Data directory contents:", os.listdir(data_path))

# Load the dataset (set download=False since you have it manually)
try:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    print(f"Successfully loaded CIFAR-10!")
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")

except RuntimeError as e:
    print(f"Error loading dataset: {e}")
    print("Please check if the dataset is properly extracted.")
