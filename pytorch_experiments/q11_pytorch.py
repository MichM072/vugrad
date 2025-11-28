import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optuna import trial
from tqdm import tqdm
import optuna

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

    def __init__(self, lr=0.001, momentum=0.9, model_name=None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = model_name if model_name is not None else 'cifar_net'

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_net(self, trainloader, save_model=False, epochs=5, trial=None):
        # Unsure if needed but added to be sure the model is set to train mode.
        self.train()
        for epoch in tqdm(range(epochs), desc='Training Epochs'):
            running_loss = 0.0
            epoch_loss_sum = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                epoch_loss_sum += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

            epoch_loss = epoch_loss_sum / len(trainloader.dataset)

            # Prevent trials that show no improvement.
            if trial is not None:
                trial.report(epoch_loss, epoch)

                if trial.should_prune():
                    print(f"Trial was pruned at epoch {epoch}")
                    raise optuna.TrialPruned()


        # Final training loss:
        # Set to eval mode to "evaluate" on the train set.
        self.eval()
        with torch.no_grad():
            for data in trainloader:
                inputs, labels = data

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)

        print('Finished Training')
        self.save_model() if save_model else None
        return loss.item() if loss.item() is not None else 0.0

    def predict(self, test_images):
        return self(test_images)

    def test_net(self, testloader, classes, load_model=False, individual_class=False):
        # prepare to count predictions for each class
        self.load_model() if load_model else None
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
        overall_accuracy = 100 * correct / total
        print(f'Overall accuracy: {overall_accuracy} %')
        return overall_accuracy


    def load_model(self, path=f'cifar_net.pth'):
        if self.model_name is not None:
            path = f'{self.model_name}.pth'
        self.load_state_dict(torch.load(path))


    def save_model(self, path='cifar_net.pth'):
        if self.model_name is not None:
            path = f'{self.model_name}.pth'
        torch.save(self.state_dict(), path)

def objective(trial):
    lr = trial.suggest_float('lr', 0.0001, 0.1, step=0.0001)
    momentum = trial.suggest_float('momentum', 0.3, 0.99, step=0.01)
    epochs = trial.suggest_int('epochs', 10, 30, step=1)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    trainloader, testloader, classes = build_data_loaders(batch_size=batch_size)
    net = NeuralNet(lr, momentum)
    return net.train_net(trainloader, epochs=epochs, trial=trial)

def final_evaluation(param_dict):
    trainloader, testloader, classes = build_data_loaders(batch_size=10)
    model_name = f'cifar_net_{param_dict.items()}'
    net = NeuralNet(param_dict['lr'], param_dict['momentum'], model_name=model_name)
    net.train_net(trainloader, epochs=param_dict['epochs'], save_model=True)
    return net.test_net(testloader, classes, individual_class=True)


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize',
                                storage='sqlite:///optuna.db',
                                study_name='cifar_net_q11',
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                   n_warmup_steps=10,
                                                                   interval_steps=1))
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print(f'Best params: {best_params}')
    print("Final evaluation:")
    final_score = final_evaluation(best_params)
    print(f'Final accuracy: {final_score:.2f}')

