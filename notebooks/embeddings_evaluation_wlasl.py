# %load_ext autoreload
# %autoreload 2

import sys
import os.path as op
import pandas as pd
import json
import base64

sys.path.append(op.abspath('..'))

import torch
import multiprocessing

from collections import Counter
from itertools import chain
# +
from clearml import Dataset, InputModel
from torch.utils.data import DataLoader

from datasets import SLREmbeddingDataset, collate_fn_padd
from spoter import embeddings_scatter_plot_splits
from spoter import SPOTER_EMBEDDINGS


# LOAD MODEL FROM CLEARML
# model id is obtained from ClearML UI
# overfitting_non_normalized = 3ca9ffbd1edc44c6be6179190a73c53a
# normalized = '323f291f2e5c4e8f8c60a94f1cdaebbd'
# normalized 0.5 = 0a13513e575a49f989f116f35879c3a9
model = InputModel(model_id='9fddb0d4389a4cdb867acada859a6b64')

checkpoint = torch.load(model.get_weights())

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

model = SPOTER_EMBEDDINGS(
    features=checkpoint["config_args"].vector_length,
    hidden_dim=checkpoint["config_args"].hidden_dim,
    norm_emb=checkpoint["config_args"].normalize_embeddings,
).to(device)

model.load_state_dict(checkpoint["state_dict"])

dataset_project = "Sign Language Recognition"
dataset_name = "wlasl_mapped_mediapipe_only_landmarks_25fps"
num_classes = 100
batch_size = 1
dataset_folder = (Dataset.get(dataset_project=dataset_project,
                              dataset_name=dataset_name)
                         .get_local_copy())
dataset_folder = Dataset.get(dataset_id='a2205366a6b34d12a55940cecfd7f09d').get_local_copy()

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
# -

labels_split = {}
embeddings_split = {}
splits = list(dataloaders.keys())
with torch.no_grad():
    for split, dataloader in dataloaders.items():
        labels_str = []
        embeddings = []
        for i, (inputs, labels, masks) in enumerate(dataloader):
            inputs = inputs.to(device)
            masks = masks.to(device)
            outputs = model(inputs, masks)
            for n in range(outputs.shape[0]):
                embeddings.append(outputs[n, 0].cpu().detach().numpy())
        embeddings_split[split] = embeddings

len(embeddings_split['train']), len(dfs['train'])

for split in splits:
    df = dfs[split]
    df['embeddings'] =  embeddings_split[split]


from scipy.spatial import distance_matrix
import numpy as np



use_centroids = False

for use_centroids, str_use_centroids in zip([True, False],
                                           ['Using centroids only', 'Using all embeddings']):

    df_val = dfs['val']
    df_train = dfs['train']
    if use_centroids:
        df_train = dfs['train'].groupby('labels')['embeddings'].apply(np.mean).reset_index()
    x_train = np.vstack(df_train['embeddings'])
    x_val = np.vstack(df_val['embeddings'])

    d_mat = distance_matrix(x_val, x_train, p=2)

    top5_embs = 0
    top5_classes = 0
    knn = 0
    top1 = 0

    len_val_dataset = len(df_val)
    good_samples = []

    for i in range(d_mat.shape[0]):
        true_label = df_val.loc[i, 'labels']
        labels = df_train['labels'].values
        argsort = np.argsort(d_mat[i])
        sorted_labels = labels[argsort]
        if sorted_labels[0] == true_label:
            top1 += 1
            if use_centroids:
                good_samples.append(df_val.loc[i, 'video_id'])
            else:
                good_samples.append((df_val.loc[i, 'video_id'],
                                     df_train.loc[argsort[0], 'video_id'],
                                     i,
                                     argsort[0]))


        if true_label == Counter(sorted_labels[:5]).most_common()[0][0]:
            knn += 1
        if true_label in sorted_labels[:5]:
            top5_embs += 1
        if true_label in list(dict.fromkeys(sorted_labels))[:5]:
            top5_classes += 1
        else:
            continue
            print(true_label, df_val.loc[i, 'label_name'], sorted_labels[:5])


    print(str_use_centroids)


    print(f'Top-1 accuracy: {100 * top1 / len_val_dataset : 0.2f} %')
    if not use_centroids:
        print(f'5-nn accuracy: {100 * knn / len_val_dataset : 0.2f} %')
    print(f'Top-5 embs class match: {100 * top5_embs / len_val_dataset: 0.2f} %')
    if not use_centroids:
        print(f'Top-5 class match: {100 * top5_classes / len_val_dataset: 0.2f} %')
    print('#'*32)

i_val

i_train

df_val.loc[i_val]

for row in df_train[df_train.label_name == 'thursday'].itertuples():
    display(Video(f'sample_videos/{row.video_id}.mp4'))

# +
for row in df_val[df_val.label_name == 'thursday'].itertuples():
    display(Video(f'sample_videos/{row.video_id}.mp4'))


# -

df_train.loc[i_train]

from IPython.display import Video

# +
display(Video(url=f"sample_videos/{val_id}.mp4"))
display(Video(url=f"sample_videos/{train_id}.mp4"))



# -

good_samples

df_train



