import torch


def make_data():
    """Return train and test dataloaders for MNIST."""
    train_data, train_labels = [ ], [ ]

    for i in range(9):
        train_data.append(torch.load(f"data/raw/corruptmnist/train_images_{i}.pt"))
        train_labels.append(torch.load(f"data/raw/corruptmnist/train_target_{i}.pt"))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    test_data = torch.load("data/raw/corruptmnist/test_images.pt")
    test_labels = torch.load("data/raw/corruptmnist/test_target.pt")

    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    #Normalize data to have 0 mean and std of 1
    train_data = (train_data - train_data.mean()) / train_data.std()
    test_data = (test_data - test_data.mean()) / test_data.std()

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    #Save data
    torch.save(train_dataset, "data/processed/train_dataset.pt")
    torch.save(test_dataset, "data/processed/test_dataset.pt")

if __name__ == "__main__":
    make_data()
