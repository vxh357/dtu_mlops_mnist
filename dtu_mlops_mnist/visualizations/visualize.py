import torch
import click
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


from dtu_mlops_mnist.models import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option('-model_path', prompt='Model path', default='models/test/model.pth', help='Path to the model.')
def visualize(model_path='models/test/model.pt'):
    """Visualize the given model using training data."""

    model_folder = model_path.split("/")[1]

    net = model.MyNeuralNet(784,10).to(device)
    net.load_state_dict(torch.load(model_path))

    # remove last two layers
    net = torch.nn.Sequential(*(list(net.children())[:-2]))
    net.eval()

    train_set = torch.load("data/processed/train_dataset.pt")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=500)
    image, label = next(iter(train_dataloader))
    image = image.reshape((500,1,-1))

    pred = net(image.to(device)).cpu().detach()

    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(pred[:,0,:])

    sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=label,
    palette=sns.color_palette("hls", 10),
    alpha=0.6
    )
    plt.title(f"t-SNE for model from {model_folder}")
    plt.legend()

    plt.savefig(f"reports/figures/{model_folder}/tsne.pdf")



if __name__ == """__main__""":
    visualize()
