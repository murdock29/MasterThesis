import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from src.networks.resnet import resnet34


class MemoryDataset(Dataset):
    """Characterizes a datasets for PyTorch -- this datasets pre-loads all images in memory"""

    def __init__(self, data, transform):
        """Initialization"""
        self.masks = data['y']
        self.images = data['x']
        unique_labels = np.unique(self.masks)  # Get unique labels
        self.masks = {label: i for i, label in enumerate(unique_labels)}
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.images[index].astype(np.float32) / 255.0
        x = self.transform(x)
        # x = x.transpose(2, 0, 1)
        y = torch.tensor(self.masks[index])
        return x, y  # we need to add the extra dimension in front again

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, 1024)
        self.fc9 = nn.Linear(1024, 1024)
        self.fc10 = nn.Linear(1024, 1024)
        self.fc11 = nn.Linear(1024, 1024)
        self.fc12 = nn.Linear(1024, 1024)
        self.fc13 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, layer=None):
        x = x.view(x.size(0), -1)
        features = []
        x = self.relu(self.fc1(x))
        features.append(x)
        x = self.relu(self.fc2(x))
        features.append(x)
        x = self.relu(self.fc3(x))
        features.append(x)
        x = self.relu(self.fc4(x))
        features.append(x)
        x = self.relu(self.fc5(x))
        features.append(x)
        x = self.relu(self.fc6(x))
        features.append(x)
        x = self.relu(self.fc7(x))
        features.append(x)
        x = self.relu(self.fc8(x))
        features.append(x)
        x = self.relu(self.fc9(x))
        features.append(x)
        x = self.relu(self.fc10(x))
        features.append(x)
        x = self.relu(self.fc11(x))
        features.append(x)
        x = self.relu(self.fc12(x))
        features.append(x)
        x = self.fc13(x)
        features.append(x)

        if layer is not None:
            return features[layer]
        return x, features


# np.random.seed(0)
# # Load CIFAR-10 and CIFAR-100 datasets
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# cifar10_trainset = torchvision.datasets.CIFAR10(root='../data/CIFAR-10/train', train=True, download=True, transform=transform)
# cifar10_trainloader = DataLoader(cifar10_trainset, batch_size=128, shuffle=True, num_workers=2)
#
# cifar100_trainset = torchvision.datasets.CIFAR100(root='../data/CIFAR-100/', train=True, download=True, transform=transform)
#
# # Select a subset of CIFAR-100 (e.g., 10 classes)
# random_classes = list(np.random.randint(0, 100, 10))
# subset_indices = np.where(np.isin(cifar100_trainset.targets, random_classes))[0]
# cifar100_subset = Subset(cifar100_trainset, subset_indices)
# cifar100_loader = DataLoader(cifar100_subset, batch_size=128, shuffle=True, num_workers=2)
#
# # Initialize the model, loss function, and optimizer
# model = MLP(num_classes=10)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.0, weight_decay=0)

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

            # todo mlp
            # outputs, _ = model(inputs)
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

# trained_model = train_model(model, cifar10_trainloader, criterion, optimizer, num_epochs=1000)


def linear_probing(model, dataloader, num_layers=12, num_classes=10):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    accuracies = []

    for layer in range(num_layers):
        features = []
        labels = []

        with torch.no_grad():
            for data in dataloader:
                inputs, batch_labels = data[0].to(device), data[1].to(device)
                layer_features = model(inputs, layer=layer)
                features.append(layer_features.cpu())
                labels.append(batch_labels.cpu())

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        # Define a linear classifier
        linear_classifier = nn.Linear(features.size(1), num_classes).to(device)
        linear_optimizer = optim.Adam(linear_classifier.parameters(), lr=0.001)
        linear_criterion = nn.CrossEntropyLoss()

        # Train the linear classifier
        for epoch in range(30):
            linear_optimizer.zero_grad()
            outputs = linear_classifier(features.to(device))
            loss = linear_criterion(outputs, labels.to(device))
            loss.backward()
            linear_optimizer.step()

        # Evaluate the linear classifier
        with torch.no_grad():
            outputs = linear_classifier(features.to(device))
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.to(device)).sum().item()
            accuracy = correct / labels.size(0)
            accuracies.append(accuracy)
            print(f'Layer {layer + 1}, Accuracy: {accuracy:.2f}')

    return accuracies

# # Perform linear probing on CIFAR-10
# cifar10_accuracies = linear_probing(trained_model, cifar10_trainloader, num_layers=12, num_classes=10)
#
# # Perform linear probing on CIFAR-100 subset
# cifar100_accuracies = linear_probing(trained_model, cifar100_loader, num_layers=12, num_classes=10)


def compute_covariance_matrix(features):
    # Center the data
    features_centered = features - np.mean(features, axis=0)
    # Compute the covariance matrix
    cov_matrix = np.cov(features_centered, rowvar=False)
    return cov_matrix

def compute_singular_values(cov_matrix):
    _, singular_values, _ = np.linalg.svd(cov_matrix)
    return singular_values

def compute_numerical_rank(singular_values, threshold=1e-3):
    max_singular_value = np.max(singular_values)
    significant_singular_values = singular_values[singular_values > threshold * max_singular_value]
    numerical_rank = len(significant_singular_values)
    return numerical_rank




if __name__ == '__main__':
    np.random.seed(0)
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
        num_workers=2,
        collate_fn=remap_labels  # Use custom collate function
    )

    # Initialize the model, loss function, and optimizer
    model = resnet34(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    # todo mlp
    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.0, weight_decay=0)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)



    # trained_model = train_model(model, cifar10_trainloader, criterion, optimizer, num_epochs=1000)
    #
    trained_model = train_model(model, cifar10_trainloader, criterion, optimizer, num_epochs=164)
    torch.save(trained_model.state_dict(), '../model/3_2_OOD/resnet_model.pth')
    # trained_model = MLP(num_classes=10)
    # trained_model.load_state_dict(torch.load('../model/3_2_OOD/mlp_model.pth'))
    # Perform linear probing on CIFAR-10
    cifar10_accuracies = linear_probing(trained_model, cifar10_trainloader, num_layers=12, num_classes=10)

    # Perform linear probing on CIFAR-100 subset
    # cifar100_accuracies = linear_probing(trained_model, cifar100_loader, num_layers=12, num_classes=10)