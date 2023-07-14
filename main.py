# Load the data
import torchvision.transforms
import multiprocessing
from utils import get_train_val_data_loaders, get_test_data_loader, show_sample_of_images, optimize
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

num_workers = multiprocessing.cpu_count()
    # print(f' num workers = {num_workers}')

# batch_size to use
batch_size = 20

# validation set size out of training dataset
validation_size = 0.2

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Load training and validation data
train_dataloader, val_dataloader = get_train_val_data_loaders(batch_size=batch_size, valid_size=validation_size, transforms=transforms, num_workers=num_workers)
test_dataloader = get_test_data_loader(batch_size=batch_size, transforms=transforms, num_workers=num_workers)

# Create a dictionary of data loaders for easy use
dataloaders = {
    'train' :train_dataloader,
    'val' : val_dataloader,
    'test' : test_dataloader
}

classes = [
    "airplane","automobile","bird","cat", "deer","dog","frog","horse","ship","truck"
]

# helper function to un-normalize and display an image
show_sample_of_images(dataloaders['train'])

class MyNet(nn.Module):
    # Implement Sequential later
    def __init__ (self, num_classes=10):
        super(MyNet, self).__init__()

        # Common activation function to be shared by layers
        self.relu = nn.ReLU()

        # convolution layer 1
        # Initial image matrix size is 32 X 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Image size =16 X 16 X 16

        # convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Image size = 32 X 8 X 8

        # convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Image size = 64 X 4 X 4

        # Linear input layer
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64 * 4 * 4, 500)
        self.dropout1 = nn.Dropout(0.2)

        self.linear2 = nn.Linear(500, num_classes)

    def forward(self, inputs):

        inputs = self.relu(self.pool1(self.conv1(inputs)))
        inputs = self.relu(self.pool2(self.conv2(inputs)))
        inputs = self.relu(self.pool3(self.conv3(inputs)))

        inputs = self.flatten(inputs)

        inputs = self.relu(self.dropout1(self.linear1(inputs)))
        inputs = self.linear2(inputs)

        return inputs

model = MyNet()

# create optimizer. Experiment with other optimizer functions like Adam
optimizer = optim.SGD(model.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()

optimize(dataloaders=dataloaders,model=model, optim=optimizer, criterion= criterion, total_epochs=20, save_path='cifar10_valid.pt')


# Test with test dataset