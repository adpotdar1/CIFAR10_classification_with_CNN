import torch
from torchvision import datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_train_val_data_loaders(batch_size, valid_size, transforms, num_workers):
    #Get the CIFAR10 dataset and apply transforms
    train_dataset =  datasets.CIFAR10('data', train=True, download= False, transform=transforms)

    # Get the length of training and validation set
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # split the data
    train_split_dataset, val_split_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Get the training and validation loaders
    train_loader = torch.utils.data.DataLoader(train_split_dataset, batch_size= batch_size, shuffle=True)      #, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_split_dataset, batch_size=batch_size, shuffle=True)           #, num_workers=num_workers)

    return train_loader, val_loader

def get_test_data_loader(batch_size, transforms, num_workers):

    # Get CIFAR10 test dataset
    test_dataset = datasets.CIFAR10('data', train=False, download=False, transform=transforms)

    # get the data loader
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle= True)         #, num_workers=num_workers)

    return test_data_loader

def train_one_epoch(train_loader, model, optim, criterion ):
    if torch.cuda.is_available():
        torch.cuda()

    # get the model in training mode
    model.train()

    train_loss = 0;

    for batch_index, (data, label) in tqdm(
        enumerate(train_loader),
        desc='Training',
        total= len(train_loader),
        leave=True,
        ncols=80
    ):

        # Stop grad calculation
        optim.zero_grad()

        # Run the model ones
        output = model(data)

        # Calculate loss
        loss_value = criterion(output, label)

        # Calculate gradients by a backward pass
        loss_value.backward()

        # Update weights
        optim.step()

        # update average training loss
        train_loss = train_loss + (
                (1 / (batch_index + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def validate_one_epoch( val_loaders, model, criterion):

    with torch.no_grad():

        if(torch.cuda.is_available()):
            torch.cuda()

        # set the model in evaluation mode
        model.eval()

        # validation loss
        val_loss = 0

        for batch_index, (data, labels) in tqdm(
            enumerate(val_loaders),
            desc= 'Validation',
            total= len(val_loaders),
            leave=True,
            ncols=80
        ):

            output = model(data)

            loss_value = criterion(output, labels)

            # Calculate average validation loss
            val_loss = val_loss + (
                    (1 / (batch_index + 1)) * (loss_value.data.item() - val_loss)
            )

        return val_loss


def optimize(dataloaders, model, optim, criterion,  total_epochs, save_path):

    min_val_loss = None
    logs = {}

    for epoch in range(1, total_epochs+ 1):

        train_loss = train_one_epoch(dataloaders['train'], model, optim, criterion)

        val_loss = validate_one_epoch(dataloaders['val'], model, criterion)

        if min_val_loss == None or (min_val_loss - val_loss)/min_val_loss >= 0.01:
            print(f"New minimum validation loss: {val_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)  # -

            valid_loss_min = val_loss

# This method shows us all images of one batch.
def show_sample_of_images( data_loader):
    classes = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]

    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    images = images.numpy()  # convert images to numpy for display

    # plot the images in the batch, along with the corresponding labels
    # display 20 images
    # NOTE: make sure your batch size is at least 20
    fig, subs = plt.subplots(2, 10, figsize=(25, 4))
    for i, sub in enumerate(subs.flatten()):
        imshow(images[i], sub)
        sub.set_title(classes[labels[i]])

    plt.show()


def imshow(img, sub):
    img = img / 2 + 0.5  # unnormalize
    sub.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    sub.axis("off")

