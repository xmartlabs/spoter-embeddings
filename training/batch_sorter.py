import logging
from datetime import datetime
import numpy as np
from typing import Optional
from .batching_scheduler import BatchingScheduler
import torch

logger = logging.getLogger("BatchGrouper")


class BatchGrouper:
    """
    Will cluster all `total_items` into `max_groups` clusters based on distances in
    `sorted_dists`. Each group has `mini_batch_size` elements and these elements are just integers in
    range 0...total_items.

    Distances between these items are expected to be scaled to 0...1 in a way that distances for two items in the
    same class are higher if closer to 1, while distances between elements of different classes are higher if closer
    to 0.

    The logic is picking the highest value distance and assigning both items to the same cluster/group if possible.
    This might include merging 2 clusters.
    There are a few threshold to limit the computational cost. If the scaled distance between a pair is below
    `dist_threshold`, or more than `assign_threshold` percent of items have been assigned to the groups, we stop and
    assign the remanining items to the groups that have space left.
    """
    # Counters
    next_group = 0
    items_assigned = 0

    # Thresholds
    dist_threshold = 0.5
    assign_threshold = 0.80

    def __init__(self, sorted_dists, total_items, mini_batch_size=32, dist_threshold=0.5, assign_threshold=0.8) -> None:
        self.sorted_dists = sorted_dists
        self.total_items = total_items
        self.mini_batch_size = mini_batch_size
        self.max_groups = int(total_items / mini_batch_size)
        self.groups = {}
        self.item_to_group = {}
        self.items_assigned = 0
        self.next_group = 0
        self.dist_threshold = dist_threshold
        self.assign_threshold = assign_threshold

    def cluster_items(self):
        """Main function of this class. Does the clustering explained in class docstring.

        :raises e: _description_
        :return _type_: _description_
        """
        for i in range(self.sorted_dists.shape[-1]):  # and some other conditions are unmet
            a, b, dist = self.sorted_dists[:, i]
            a, b = int(a), int(b)
            if dist < self.dist_threshold or self.items_assigned > self.total_items * self.assign_threshold:
                logger.info(f"Breaking with dist: {dist}, and {self.items_assigned} items assigned")
                break
            if a not in self.item_to_group and b not in self.item_to_group:
                g = self.create_or_get_group()
                self.assign_group(a, g)
                self.assign_group(b, g)
            elif a not in self.item_to_group:
                if not self.group_is_full(self.item_to_group[b]):
                    self.assign_group(a, self.item_to_group[b])
            elif b not in self.item_to_group:
                if not self.group_is_full(self.item_to_group[a]):
                    self.assign_group(b, self.item_to_group[a])
            else:
                grp_a = self.item_to_group[a]
                grp_b = self.item_to_group[b]
                self.merge_groups(grp_a, grp_b)
        self.assign_remaining_items()
        return list(np.concatenate(list(self.groups.values())).flat)

    def assign_group(self, item, group):
        """Assigns `item` to group `group`
        """
        self.item_to_group[item] = group
        self.groups[group].append(item)
        self.items_assigned += 1

    def create_or_get_group(self):
        """Creates a new group if current group count is less than max_groups.
        Otherwise returns first group with space left.

        :return int: The group id
        """
        if self.next_group < self.max_groups:
            group = self.next_group
            self.groups[group] = []
            self.next_group += 1
        else:
            for i in range(self.next_group):
                if len(self.groups[i]) <= self.mini_batch_size - 2:
                    group = i
                    break  # out of the for loop
        return group

    def group_is_full(self, group):
        return len(self.groups[group]) == self.mini_batch_size

    def can_merge_groups(self, grp_a, grp_b):
        return grp_a != grp_b and (len(self.groups[grp_a]) + len(self.groups[grp_b]) < self.mini_batch_size)

    def merge_groups(self, grp_a, grp_b):
        """Will merge two groups together, if possible. Otherwise does nothing.
        """
        if grp_a > grp_b:
            grp_a, grp_b = grp_b, grp_a
        if self.can_merge_groups(grp_a, grp_b):
            logger.debug(f"MERGE {grp_a} with {grp_b}: {len(self.groups[grp_a])} {len(self.groups[grp_b])}")
            for b in self.groups[grp_b]:
                self.item_to_group[b] = grp_a
            self.groups[grp_a].extend(self.groups[grp_b])
            self.groups[grp_b] = []
            self.replace_group(grp_b)

    def replace_group(self, group):
        """Replace a group with the last one in the list

        :param int group: Group to replace
        """
        grp_to_change = self.next_group - 1
        if grp_to_change != group:
            for item in self.groups[grp_to_change]:
                self.item_to_group[item] = group
            self.groups[group] = self.groups[grp_to_change]
        del self.groups[grp_to_change]
        self.next_group -= 1

    def assign_remaining_items(self):
        """ Assign remaining items into groups
        """
        grp_pointer = 0
        i = 0
        logger.info(f"Assigning rest of items: {self.items_assigned} of {self.total_items}")
        while i < self.total_items:
            if i not in self.item_to_group:
                if grp_pointer not in self.groups:
                    # This would happen if a group is still empty at this stage
                    assert grp_pointer < self.max_groups
                    new_group = self.create_or_get_group()
                    assert new_group == grp_pointer
                if len(self.groups[grp_pointer]) < self.mini_batch_size:
                    self.assign_group(i, grp_pointer)
                    i += 1
                else:
                    grp_pointer += 1
            else:
                i += 1


def get_dist_tuple_list(dist_matrix):
    batch_size = dist_matrix.size()[0]
    indices = torch.tril_indices(batch_size, batch_size, offset=-1)
    values = dist_matrix[indices[0], indices[1]].cpu()
    return torch.cat([indices, values.unsqueeze(0)], dim=0)


def get_scaled_distances(embeddings, labels, device, same_label_factor=1):
    """Returns distance matrix between all embeddings scaled to the 0-1 range where 0 is good and 1 is bad.
    This means that small distances for embeddings of the same class will be close to 0 while small distances for
    embeddings of different classes will be close to 1

    :param _type_ embeddings: Embeddings of batch items
    :param _type_ labels: Labels associated to the embeddings
    :param _type_ device: Device to run on (cuda or cpu)
    :param int same_label_factor: Multiplies the weight of same-class distances allowing to give more or less importance
    to these compared to distinct-class distances, defaults to 1 (which means equal weight)
    :return torch.Tensor: Scaled distance matrix
    """
    # Get pairwise distance matrix
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    # Get list of tuples with emb_A, emb_B, dist ordered by greater for same label and smaller for diff label
    # shape: (batch_size, batch_size)
    labels = labels.to(device)
    labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1)).squeeze()
    labels_distinct = torch.logical_not(labels_equal)
    pos_dist = distance_matrix * labels_equal
    neg_dist = distance_matrix * labels_distinct

    # Use some scaling to bring both to a range of 0-1
    pos_max = pos_dist.max()
    neg_max = neg_dist.max()
    # Closer to 1 is harder
    pos_dist = pos_dist / pos_max * same_label_factor
    neg_dist = 1 * labels_distinct - (neg_dist / neg_max)
    return pos_dist + neg_dist


def sort_batches(inputs, labels, masks, embeddings, device, mini_batch_size=32,
                 scheduler: Optional[BatchingScheduler] = None):
    start = datetime.now()

    same_label_factor = scheduler.get_scaling_same_label_factor() if scheduler else 1
    scaled_dist = get_scaled_distances(embeddings, labels, device, same_label_factor)
    # Get vector of (row, column, dist)
    dist_list = get_dist_tuple_list(scaled_dist)

    dist_list = dist_list.cpu().detach().numpy()
    # Sort distances descending by last row
    sorted_dists = dist_list[:, dist_list[-1, :].argsort()[::-1]]

    # Loop through list assigning both items to same group
    dist_threshold = scheduler.get_dist_threshold() if scheduler else 0.5
    grouper = BatchGrouper(sorted_dists, total_items=labels.size()[0], mini_batch_size=mini_batch_size,
                           dist_threshold=dist_threshold)
    indices = torch.tensor(grouper.cluster_items()).type(torch.IntTensor)
    final_inputs = torch.index_select(inputs, dim=0, index=indices)
    final_labels = torch.index_select(labels, dim=0, index=indices)
    final_masks = torch.index_select(masks, dim=0, index=indices)

    logger.info(f"Batch sorting took: {datetime.now() - start}")
    return final_inputs, final_labels, final_masks
