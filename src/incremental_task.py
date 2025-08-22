import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from torchvision.models import vgg19_bn

from src.networks.resnet import resnet34

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

# Training loop
def train_task(model, dataloader, num_epochs=30, lr=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)

        print(f'Epoch {epoch + 1}, Loss: {running_loss / total:.3f}')
        scheduler.step()
    return model

def evaluate_task(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Split CIFAR-10 into two tasks
def split_cifar10(full_dataset, task_classes):
    task_datasets = []
    for classes in task_classes:
        indices = [i for i, label in enumerate(full_dataset.targets) if label in classes]
        task_dataset = Subset(full_dataset, indices)
        task_datasets.append(task_dataset)
    return task_datasets

def extractor_tunnel_analysis(model, task_loaders):
    # Assume the first 8 layers are the extractor and the rest are the tunnel
    extractor = nn.Sequential(*list(model.children())[0][:7])
    tunnel_classifier = list(model.children())[1]

    # Freeze the extractor and train a new classifier on Task 2
    for param in extractor.parameters():
        param.requires_grad = False

    new_model = nn.Sequential(extractor, nn.Flatten(), tunnel_classifier)
    new_model = train_task(new_model, task_loaders[1])

    acc_task1 = evaluate_task(new_model, task_loaders[0])
    acc_task2 = evaluate_task(new_model, task_loaders[1])

    print(f"Accuracy on Task 1 with frozen extractor: {acc_task1:.4f}")
    print(f"Accuracy on Task 2 with frozen extractor: {acc_task2:.4f}")



if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    # Load CIFAR-10 and CIFAR-100 datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar10_trainset = torchvision.datasets.CIFAR10(root='../data/CIFAR-10/train', train=True, download=True,
                                                    transform=transform)
    full_set = range(10)
    random_classes = list(np.random.choice(full_set, 7, replace=False))
    subset = [x for x in full_set if x not in random_classes]
    task_classes = [random_classes, subset]
    task_datasets = split_cifar10(cifar10_trainset, task_classes)
    task_loaders = [DataLoader(dataset, batch_size=128, shuffle=True) for dataset in task_datasets]

    # Initialize the model, loss function, and optimizer
    model = vgg19_bn()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


    for i, task in enumerate(task_loaders):
        print(f"Training Task {i+1}...")
        trained_model = train_task(model, task, num_epochs=160)
        torch.save(trained_model.state_dict(), f'../model/4_incremental_task/task1_high_task{i+1}_model.pth')
        evaluate_task(trained_model, task)
    #
    # trained_model = train_model(model, cifar10_trainloader, criterion, optimizer, num_epochs=164)
    # torch.save(trained_model.state_dict(), '../model/3_2_OOD/resnet_model.pth')
    # trained_model = resnet34(num_classes=10)
    # trained_model.load_state_dict(torch.load('../model/3_2_OOD/resnet_model.pth'))
