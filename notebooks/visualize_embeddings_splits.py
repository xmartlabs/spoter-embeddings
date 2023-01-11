# %load_ext autoreload
# %autoreload 2

import os
import sys
import os.path as op
import pandas as pd
import json
import base64

sys.path.append(op.abspath('..'))

import torch
import multiprocessing

from itertools import chain
# +
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import LinearColorMapper, ColumnDataSource
from torch.utils.data import DataLoader

from datasets import SLREmbeddingDataset, collate_fn_padd
from datasets.dataset_loader import LocalDatasetLoader
from models import embeddings_scatter_plot_splits
from models import SPOTER_EMBEDDINGS


BASE_DATA_FOLDER = 'data/'
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# LOAD MODEL FROM CLEARML
# from clearml import InputModel
# model = InputModel(model_id='9fddb0d4389a4cdb867acada859a6b64')
# checkpoint = torch.load(model.get_weights())


CHECKPOINT_PATH = "checkpoints/checkpoint.pth"
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)


model = SPOTER_EMBEDDINGS(
    features=checkpoint["config_args"].vector_length,
    hidden_dim=checkpoint["config_args"].hidden_dim,
    norm_emb=checkpoint["config_args"].normalize_embeddings,
).to(device)

model.load_state_dict(checkpoint["state_dict"])


def get_dataset_loader(loader_name=None):
    if loader_name == 'CLEARML':
        from datasets.clearml_dataset_loader import ClearMLDatasetLoader
        return ClearMLDatasetLoader()
    else:
        return LocalDatasetLoader()

dataset_loader = get_dataset_loader()
dataset_project = "Sign Language Recognition"
dataset_name = "wlasl_mapped_mediapipe_only_landmarks_25fps"
num_classes = 100
batch_size = 1
dataset_folder = dataset_loader.get_dataset_folder(dataset_project, dataset_name)

# +
dataloaders = {}
splits = ['train', 'val']
dfs = {}
for split in splits:
    split_set_path = op.join(dataset_folder, f"WLASL{num_classes}_{split}_25fps.csv")
    split_set = SLREmbeddingDataset(split_set_path, triplet=False)
    data_loader = DataLoader(
        split_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_padd,
        pin_memory=torch.cuda.is_available(),
        num_workers=multiprocessing.cpu_count()
    )
    dataloaders[split] = data_loader
    dfs[split] =  pd.read_csv(split_set_path)

with open(op.join(dataset_folder, 'id_to_label.json')) as fid:
    id_to_label = json.load(fid)
id_to_label = {int(key): value for key, value in id_to_label.items()}

tsne_results, labels_results = embeddings_scatter_plot_splits(model,
                                                              dataloaders,
                                                              device,
                                                              id_to_label,
                                                              perplexity=40,
                                                              n_iter=1000)


set_labels = list(set(next(chain(labels_results.values()))))

# +
dfs = {}
for split in splits:
    split_set_path = op.join(dataset_folder, f"WLASL{num_classes}_{split}_25fps.csv")
    df =  pd.read_csv(split_set_path)
    df['tsne_x'] = tsne_results[split][:, 0]
    df['tsne_y'] = tsne_results[split][:, 1]
    df['split'] = split
    df['video_fn'] = df['video_id'].apply(lambda video_id: os.path.join(BASE_DATA_FOLDER, f'wlasl/videos/{video_id:05d}.mp4'))
    dfs[split] = df

output_notebook()

df = pd.concat([dfs['train'].sample(100), dfs['val']]).reset_index(drop=True)
len(df)
# -



# +
from tqdm.auto import tqdm

def load_videos(video_list):
    print('loading videos')
    videos = []
    for video_fn in tqdm(video_list):
        if video_fn is None:
            video_data = None
        else:
            with open(video_fn, 'rb') as fid:
                video_data = base64.b64encode(fid.read()).decode()
        videos.append(video_data)
    print('Done loading videos')
    return videos


# +

use_img_div = True

img_div = '''
        <div>
            <img
                src="data:video/mp4;base64,@videos" height="90" alt="@videos" width="120"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
'''
TOOLTIPS = f"""
    <div>
        {img_div if use_img_div else ''}
        <div>
            <span style="font-size: 17px; font-weight: bold;">@label_desc - @split</span>
            <span style="font-size: 15px; color: #966;">[$index]</span>
        </div>
        </div>
    </div>
"""
cmap = LinearColorMapper(palette="Turbo256", low=0, high=len(set_labels))
p = figure(width=1000,
           height=800,
           tooltips=TOOLTIPS,
           title=f"Check {'video' if use_img_div else 'label'} by hovering mouse over the dots")

emb_videos = load_videos(df['video_fn'])

# +
source = ColumnDataSource(data=dict(
    x=df['tsne_x'],
    y=df['tsne_y'],
    label=df['labels'],
    label_desc=df['label_name'],
    split=df['split'],
    videos=emb_videos,
))

p.scatter('x', 'y',
     size=10,
     source=source,
     fill_color={"field": 'label', "transform": cmap},
     line_color={"field": 'label', "transform": cmap}
         )
     #legend_label={"field": 'split', "transform": lambda x: df['split']},
     #marker={"field": 'split'})

show(p)
# -

df
