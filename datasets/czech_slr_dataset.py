import torch
import numpy as np
import torch.utils.data as torch_data

from datasets.datasets_utils import load_dataset, tensor_to_dictionary, dictionary_to_tensor, \
    random_augmentation
from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict


class CzechSLRDataset(torch_data.Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]
    labels: [np.ndarray]

    def __init__(self, dataset_filename: str, num_labels=5, transform=None, augmentations=False,
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
        self.num_labels = num_labels
        self.transform = transform

        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """

        depth_map = torch.from_numpy(np.copy(self.data[idx]))
        # label = torch.Tensor([self.labels[idx] - 1])
        label = torch.Tensor([self.labels[idx]])

        depth_map = tensor_to_dictionary(depth_map)

        # Apply potential augmentations
        depth_map = random_augmentation(self.augmentations, self.augmentations_prob, depth_map)

        if self.normalize:
            depth_map = normalize_single_body_dict(depth_map)
            depth_map = normalize_single_hand_dict(depth_map)

        depth_map = dictionary_to_tensor(depth_map)

        # Move the landmark position interval to improve performance
        depth_map = depth_map - 0.5

        if self.transform:
            depth_map = self.transform(depth_map)

        return depth_map, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    pass
