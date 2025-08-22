import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from src.networks.resnet import resnet34

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

layer_names = [
    'relu',
    'layer1.0.relu',
    'layer1.0.relu_1',
    'layer1.1.relu',
    'layer1.1.relu_1',
    'layer1.2.relu',
    'layer1.2.relu_1',
    'layer2.0.relu',
    'layer2.0.relu_1',
    'layer2.1.relu',
    'layer2.1.relu_1',
    'layer2.2.relu',
    'layer2.2.relu_1',
    'layer2.3.relu',
    'layer2.3.relu_1',
    'layer3.0.relu',
    'layer3.0.relu_1',
    'layer3.1.relu',
    'layer3.1.relu_1',
    'layer3.2.relu',
    'layer3.2.relu_1',
    'layer3.3.relu',
    'layer3.3.relu_1',
    'layer3.4.relu',
    'layer3.4.relu_1',
    'layer3.5.relu',
    'layer3.5.relu_1',
    'layer4.0.relu',
    'layer4.0.relu_1',
    'layer4.1.relu',
    'layer4.1.relu_1',
    'layer4.2.relu',
    'layer4.2.relu_1'
]
# Training loop
def train_model(model, trainloader, criterion, optimizer, num_epochs=1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 123], gamma=0.1)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} completed')
        scheduler.step()

    print('Finished Training')
    return model



def linear_probing(model, dataloader, num_layers=12, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    nodes, _ = get_graph_node_names(model)
    print(nodes)
    accuracies = []
    feature_extractor = create_feature_extractor(
        model, return_nodes=layer_names)


    for layer_num, layer in enumerate(layer_names):
        features = []
        labels = []
        with torch.no_grad():
            for data in dataloader:
                inputs, batch_labels = data[0].to(device), data[1].to(device)
                layer_features = feature_extractor(inputs)
                features.append(layer_features)
                labels.append(batch_labels)

        # og_dict = {}
        # https: // stackoverflow.com / questions / 5946236 / how - to - merge - dicts - collecting - values -from-matching - keys
        # for dict in features:
        #     for key in dict.keys():
        #         og_dict[key] = torch.cat(list(og_dict[key] for og_dict in features), dim=0)


        features = torch.cat(list(og_dict[layer] for og_dict in features), dim=0)
        # features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        features = features.view(features.size(0), -1)
        # Define a linear classifier
        linear_classifier = nn.Linear(features.size(1), num_classes).to(device)
        linear_optimizer = optim.Adam(linear_classifier.parameters(), lr=0.001)
        linear_criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Train the linear classifier
        for epoch in range(30):
            linear_optimizer.zero_grad()
            outputs = linear_classifier(features.to(device))
            loss = linear_criterion(outputs, labels.to(device))
            loss.backward()
            linear_optimizer.step()

        # Evaluate the linear classifier
        numerical_rank, sing_values = calculate_numerical_rank(features, threshold=1e-3)
        with torch.no_grad():
            outputs = linear_classifier(features.to(device))
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.to(device)).sum().item()
            accuracy = correct / labels.size(0)
            accuracies.append(accuracy)
            print(f'Layer {layer_num + 1}, Accuracy: {accuracy:.2f}, Numerical Rank: {numerical_rank}')
        # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    return accuracies

def compute_covariance_matrix(features):
    # Center the data
    # features_centered = features - torch.mean(features, dim=0, keepdim=True)
    # # Compute the covariance matrix
    # cov_matrix = torch.mm(features_centered.T, features_centered) / (features.size(0) - 1)
    cov_matrix = torch.cov(features.T)

    return cov_matrix

def compute_singular_values(cov_matrix):
    """
    Compute singular values using SVD with PyTorch.
    """
    # Perform SVD on the GPU
    _, singular_values, _ = torch.svd(cov_matrix)
    return singular_values

def compute_numerical_rank(singular_values, threshold=1e-3):
    """
    Compute the numerical rank based on singular value thresholding.
    """
    max_singular_value = torch.max(singular_values)
    # Identify significant singular values
    significant_singular_values = singular_values[singular_values > threshold * max_singular_value]
    numerical_rank = significant_singular_values.size(0)
    return numerical_rank

# def compute_covariance_matrix(features):
#     # Center the data
#     features_centered = features - np.mean(features, axis=0)
#     # Compute the covariance matrix
#     cov_matrix = np.cov(features_centered, rowvar=False)
#     return cov_matrix
#
# def compute_singular_values(cov_matrix):
#     _, singular_values, _ = np.linalg.svd(cov_matrix)
#     return singular_values
#
# def compute_numerical_rank(singular_values, threshold=1e-3):
#     max_singular_value = np.max(singular_values)
#     significant_singular_values = singular_values[singular_values > threshold * max_singular_value]
#     numerical_rank = len(significant_singular_values)
#     return numerical_rank
import time
def calculate_numerical_rank(features, threshold=1e-3):
    # Compute sample covariance matrix
    start = time.time()
    sample_cov_matrix = compute_covariance_matrix(features)
    first_time = time.time() - start
    # Compute singular values
    singular_values = compute_singular_values(sample_cov_matrix)
    second_time = time.time() - start - first_time
    # Compute numerical rank
    numerical_rank = compute_numerical_rank(singular_values, threshold)
    third_time = time.time() - start - first_time - second_time
    print(f"First Time: {first_time}, Second Time: {second_time}, Third Time: {third_time}")
    return numerical_rank, singular_values




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
    cifar10_trainloader = DataLoader(cifar10_trainset, batch_size=128, shuffle=True, num_workers=2)

    cifar100_trainset = torchvision.datasets.CIFAR100(root='../data/CIFAR-100/', train=True, download=True,
                                                      transform=transform)

    # Select a subset of CIFAR-100 (e.g., 10 classes)
    random_classes = list(np.random.randint(0, 100, 10))
    # subset_indices = np.where(np.isin(cifar100_trainset.targets, random_classes))[0]
    # cifar100_subset = Subset(cifar100_trainset, subset_indices)
    # cifar100_loader = DataLoader(cifar100_subset, batch_size=512, shuffle=True, num_workers=2)

    # Map labels to a contiguous range (0-9)
    class_map = {cls: i for i, cls in enumerate(random_classes)}  # Create mapping for random classes

    # Get subset indices where labels match the random classes
    subset_indices = np.where(np.isin(cifar100_trainset.targets, random_classes))[0]
    cifar100_subset = Subset(cifar100_trainset, subset_indices)  # Subset the dataset


    # Define a custom collate function to remap the labels
    def remap_labels(batch):
        images, labels = zip(*batch)  # Batch contains tuples of (image, label)
        labels = torch.tensor([class_map[label] for label in labels])  # Remap labels
        return torch.stack(images), labels


    # Create DataLoader with the custom collate_fn for remapped labels
    cifar100_loader = DataLoader(
        cifar100_subset,
        batch_size=512,
        shuffle=True,
        num_workers=0,
        collate_fn=remap_labels  # Use custom collate function
    )

    # Initialize the model, loss function, and optimizer
    model = resnet34(num_classes=10)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)



    # trained_model = train_model(model, cifar10_trainloader, criterion, optimizer, num_epochs=1000)
    #
    # trained_model = train_model(model, cifar10_trainloader, criterion, optimizer, num_epochs=164)
    # torch.save(trained_model.state_dict(), '../model/3_2_OOD/resnet_model.pth')
    trained_model = resnet34(num_classes=10)
    trained_model.load_state_dict(torch.load('../model/3_2_OOD/resnet_model.pth'))
    # Perform linear probing on CIFAR-10
    # cifar10_accuracies = linear_probing(trained_model, cifar10_trainloader, num_layers=12, num_classes=10)

    # Perform linear probing on CIFAR-100 subset
    cifar100_accuracies = linear_probing(trained_model, cifar100_loader, num_layers=12, num_classes=10)