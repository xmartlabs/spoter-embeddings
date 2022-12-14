import pandas as pd
import ast
import torch
from torch.nn.utils.rnn import pad_sequence
from random import randrange

from augmentations import *
from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS


HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]


def load_dataset(file_location: str):

    # Load the datset csv file
    df = pd.read_csv(file_location, encoding="utf-8")

    # TO BE DELETED
    df.columns = [item.replace("_left_", "_0_").replace("_right_", "_1_") for item in list(df.columns)]
    if "neck_X" not in df.columns:
        df["neck_X"] = [0] * len(df)
        df["neck_Y"] = [0] * len(df)

    # TEMP
    labels = df["labels"].to_list()
    # labels = [label + 1 for label in df["labels"].to_list()]
    data = []

    for row_index, row in df.iterrows():
        current_row = np.empty(shape=(len(ast.literal_eval(row["leftEar_X"])),
                                      len(BODY_IDENTIFIERS + HAND_IDENTIFIERS),
                                      2)
                               )
        for index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
            current_row[:, index, 0] = ast.literal_eval(row[identifier + "_X"])
            current_row[:, index, 1] = ast.literal_eval(row[identifier + "_Y"])

        data.append(current_row)

    return data, labels


def tensor_to_dictionary(landmarks_tensor: torch.Tensor) -> dict:

    data_array = landmarks_tensor.numpy()
    output = {}

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[identifier] = data_array[:, landmark_index]

    return output


def dictionary_to_tensor(landmarks_dict: dict) -> torch.Tensor:

    output = np.empty(shape=(len(landmarks_dict["leftEar"]), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return torch.from_numpy(output)


def random_augmentation(augmentations, augmentations_prob, depth_map):
    if augmentations and random.random() < augmentations_prob:
        selected_aug = randrange(4)
        if selected_aug == 0:
            depth_map = augment_arm_joint_rotate(depth_map, 0.3, (-4, 4))
        elif selected_aug == 1:
            depth_map = augment_shear(depth_map, "perspective", (0, 0.1))
        elif selected_aug == 2:
            depth_map = augment_shear(depth_map, "squeeze", (0, 0.15))
        elif selected_aug == 3:
            depth_map = augment_rotate(depth_map, (-13, 13))

    return depth_map


def collate_fn_triplet_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # batch: list of length batch_size, each element contains ouput of dataset
    # MASKING
    anchor_lengths = [element[0].shape[0] for element in batch]
    max_anchor_l = max(anchor_lengths)
    positive_lengths = [element[1].shape[0] for element in batch]
    max_positive_l = max(positive_lengths)
    negative_lengths = [element[2].shape[0] for element in batch]
    max_negative_l = max(negative_lengths)

    anchor_mask = [[False] * anchor_lengths[n] + [True] * (max_anchor_l - anchor_lengths[n]) \
                   for n in range(len(batch))]
    positive_mask = [[False] * positive_lengths[n] + [True] * (max_positive_l - positive_lengths[n]) \
                   for n in range(len(batch))]
    negative_mask = [[False] * negative_lengths[n] + [True] * (max_negative_l - negative_lengths[n]) \
                   for n in range(len(batch))]

    # PADDING
    anchor_batch = [element[0] for element in batch]
    positive_batch = [element[1] for element in batch]
    negative_batch = [element[2] for element in batch]

    anchor_batch = pad_sequence(anchor_batch, batch_first=True)
    positive_batch = pad_sequence(positive_batch, batch_first=True)
    negative_batch = pad_sequence(negative_batch, batch_first=True)

    return anchor_batch, positive_batch, negative_batch, \
            torch.Tensor(anchor_mask), torch.Tensor(positive_mask), torch.Tensor(negative_mask)


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # batch: list of length batch_size, each element contains ouput of dataset
    # MASKING
    anchor_lengths = [element[0].shape[0] for element in batch]
    max_anchor_l = max(anchor_lengths)

    anchor_mask = [[False] * anchor_lengths[n] + [True] * (max_anchor_l - anchor_lengths[n]) \
                   for n in range(len(batch))]

    # PADDING
    anchor_batch = [element[0] for element in batch]
    anchor_batch = pad_sequence(anchor_batch, batch_first=True)

    labels = torch.Tensor([element[1] for element in batch])

    return anchor_batch, labels, torch.Tensor(anchor_mask)
