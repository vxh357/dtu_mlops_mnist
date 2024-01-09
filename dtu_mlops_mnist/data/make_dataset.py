import torch


def make_data():
    """
    Creates and saves training and testing datasets for the CorruptMNIST dataset.

    This function loads training and testing data from stored .pt files, concatenates them,
    and performs normalization. Finally, it saves the processed datasets in a specified location.

    The function assumes the data is stored in 'data/raw/' directory in the form of .pt files.
    The processed data is saved in the 'data/processed/' directory.

    No parameters are needed and no return values are provided. The function operates through file I/O.
    """
    # Initialize lists to store training data and labels
    train_data, train_labels = [], []

    # Load training data and labels from multiple files and append to lists
    for i in range(9):
        # Loading training images and labels from each file
        train_data.append(torch.load(f"data/raw/train_images_{i}.pt"))
        train_labels.append(torch.load(f"data/raw/train_target_{i}.pt"))

    # Concatenate all training data and labels along the first dimension
    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Load testing data and labels from files
    test_data = torch.load("data/raw/test_images.pt")
    test_labels = torch.load("data/raw/test_target.pt")

    # Add a channel dimension to the data tensors
    train_data = train_data.unsqueeze(1)  # Transforming shape from [N, H, W] to [N, C, H, W]
    test_data = test_data.unsqueeze(1)  # where N is batch size, C is channel (1 here), H is height, W is width

    # Normalize data to have 0 mean and standard deviation of 1
    train_data = (train_data - train_data.mean()) / train_data.std()
    test_data = (test_data - test_data.mean()) / test_data.std()

    # Creating dataset objects from the data and labels
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    # Save the processed datasets for future use
    torch.save(train_dataset, "data/processed/train_dataset.pt")
    torch.save(test_dataset, "data/processed/test_dataset.pt")


if __name__ == "__main__":
    make_data()
