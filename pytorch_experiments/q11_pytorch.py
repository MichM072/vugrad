import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def build_data_loaders(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


class NeuralNet(torch.nn.Module):
    """
    Slightly altered implementation based on https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-loss-function-and-optimizer
    """

    def __init__(self, lr=0.001, momentum=0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = 'cifar_net'

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_net(self, trainloader, epochs=5):
        for epoch in tqdm(range(epochs), label='Training Epochs'):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

    def predict(self, test_images):
        return self(test_images)

    def test_net(self, testloader, classes, individual_class=False):
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        correct = 0
        total = 0
        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predictions = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        if individual_class:
            # print accuracy for each class
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        # print overall accuracy
        print(f'Overall accuracy: {100 * correct // total} %')
        return correct // total


def load_model(self, path=f'cifar_net.pth'):
    if self.model_name is not None:
        path = f'{self.model_name}.pth'
    self.load_state_dict(torch.load(path))


def save_model(self, path='cifar_net.pth'):
    if self.model_name is not None:
        path = f'{self.model_name}.pth'
    torch.save(self.state_dict(), path)

if __name__ == '__main__':
    net = NeuralNet()
    trainloader, testloader, classes = build_data_loaders(batch_size=4)
    net.train_net(trainloader, epochs=5)
    net.test_net(testloader, classes, individual_class=True)
