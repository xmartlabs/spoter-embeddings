import torch
import torch.utils.data as torch_data
from random import sample
from typing import List
import numpy as np

from datasets.datasets_utils import load_dataset, tensor_to_dictionary, dictionary_to_tensor, \
        random_augmentation
from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict


class SLREmbeddingDataset(torch_data.Dataset):
    """Advanced object representation of the WLASL dataset for loading triplet used in triplet loss utilizing the
    Torch's built-in Dataset properties"""

    data: List[np.ndarray]
    labels: List[np.ndarray]

    def __init__(self, dataset_filename: str, triplet=True, transform=None, augmentations=False,
                 augmentations_prob=0.5, normalize=True):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """

        loaded_data = load_dataset(dataset_filename)
        data, labels = loaded_data[0], loaded_data[1]

        self.data = data
        self.labels = labels
        self.targets = list(labels)
        self.transform = transform
        self.triplet = triplet
        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """
        depth_map_a = torch.from_numpy(np.copy(self.data[idx]))
        label = torch.Tensor([self.labels[idx]])

        depth_map_a = tensor_to_dictionary(depth_map_a)

        if self.triplet:
            positive_indexes = list(np.where(np.array(self.labels) == self.labels[idx])[0])
            positive_index_sample = sample(positive_indexes, 2)
            positive_index = positive_index_sample[0] if positive_index_sample[0] != idx else positive_index_sample[1]
            negative_indexes = list(np.where(np.array(self.labels) != self.labels[idx])[0])
            negative_index = sample(negative_indexes, 1)[0]
            # TODO: implement hard triplets

            depth_map_p = torch.from_numpy(np.copy(self.data[positive_index]))
            depth_map_n = torch.from_numpy(np.copy(self.data[negative_index]))

            depth_map_p = tensor_to_dictionary(depth_map_p)
            depth_map_n = tensor_to_dictionary(depth_map_n)

        # TODO: Add Data augmentation to positive and negative ?

        # Apply potential augmentations
        depth_map_a = random_augmentation(self.augmentations, self.augmentations_prob, depth_map_a)

        if self.normalize:
            depth_map_a = normalize_single_body_dict(depth_map_a)
            depth_map_a = normalize_single_hand_dict(depth_map_a)
            if self.triplet:
                depth_map_p = normalize_single_body_dict(depth_map_p)
                depth_map_p = normalize_single_hand_dict(depth_map_p)
                depth_map_n = normalize_single_body_dict(depth_map_n)
                depth_map_n = normalize_single_hand_dict(depth_map_n)

        depth_map_a = dictionary_to_tensor(depth_map_a)
        # Move the landmark position interval to improve performance
        depth_map_a = depth_map_a - 0.5

        if self.triplet:
            depth_map_p = dictionary_to_tensor(depth_map_p)
            depth_map_p = depth_map_p - 0.5
            depth_map_n = dictionary_to_tensor(depth_map_n)
            depth_map_n = depth_map_n - 0.5

        if self.transform:
            depth_map_a = self.transform(depth_map_a)
            if self.triplet:
                depth_map_p = self.transform(depth_map_p)
                depth_map_n = self.transform(depth_map_n)

        if self.triplet:
            return depth_map_a, depth_map_p, depth_map_n

        return depth_map_a, label

    def __len__(self):
        return len(self.labels)
