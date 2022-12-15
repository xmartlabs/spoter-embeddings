import os
import random
import numpy as np
import pandas as pd
import plotly.express as px
import torch

from models import embeddings_scatter_plot, embeddings_scatter_plot_splits


def train_setup(seed, experiment_name):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def create_embedding_scatter_plots(tracker, model, train_loader, val_loader, device, id_to_label, epoch, model_name):
    tsne_results, labels = embeddings_scatter_plot(model, train_loader, device, id_to_label, perplexity=40, n_iter=1000)

    df = pd.DataFrame({'x': tsne_results[:, 0],
                       'y': tsne_results[:, 1],
                       'label': labels})
    fig = px.scatter(df, y="y", x="x", color="label")

    tracker.log_chart(
        title="Training Scatter Plot with Best Model: " + model_name,
        series="Scatter Plot",
        iteration=epoch,
        figure=fig
    )

    tsne_results, labels = embeddings_scatter_plot(model, val_loader, device, id_to_label, perplexity=40, n_iter=1000)

    df = pd.DataFrame({'x': tsne_results[:, 0],
                       'y': tsne_results[:, 1],
                       'label': labels})
    fig = px.scatter(df, y="y", x="x", color="label")

    tracker.log_chart(
        title="Validation Scatter Plot with Best Model: " + model_name,
        series="Scatter Plot",
        iteration=epoch,
        figure=fig,
    )

    dataloaders = {'train': train_loader,
                   'val': val_loader}
    splits = list(dataloaders.keys())
    tsne_results_splits, labels_splits = embeddings_scatter_plot_splits(model, dataloaders,
                                                                        device, id_to_label, perplexity=40, n_iter=1000)
    tsne_results = np.vstack([tsne_results_splits[split] for split in splits])
    labels = np.concatenate([labels_splits[split] for split in splits])
    split = np.concatenate([[split]*len(labels_splits[split]) for split in splits])
    df = pd.DataFrame({'x': tsne_results[:, 0],
                       'y': tsne_results[:, 1],
                       'label': labels,
                       'split': split})
    fig = px.scatter(df, y="y", x="x", color="label", symbol='split')
    tracker.log_chart(
        title="Scatter Plot of train and val with Best Model: " + model_name,
        series="Scatter Plot",
        iteration=epoch,
        figure=fig,
    )
