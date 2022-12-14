import unittest
# from traceback_with_variables import activate_by_import  #noqa
import torch
from training.batch_sorter import BatchGrouper, sort_batches, get_scaled_distances, get_dist_tuple_list


class TestBatchSorting(unittest.TestCase):

    def get_sorted_dists(self):
        device = get_device()
        embeddings = torch.rand(32*8, 8).to(device)
        labels = torch.rand(32*8, 1)
        scaled_dist = get_scaled_distances(embeddings, labels, device)
        # Get vector of (row, column, dist)
        dist_list = get_dist_tuple_list(scaled_dist)

        A = dist_list.cpu().detach().numpy()
        return A[:, A[-1, :].argsort()[::-1]]

    def setUp(self) -> None:
        dists = self.get_sorted_dists()
        self.grouper = BatchGrouper(sorted_dists=dists, total_items=32*8, mini_batch_size=32)
        return super().setUp()

    def test_assigns_and_merges(self):
        group0 = self.grouper.create_or_get_group()
        self.grouper.assign_group(1, group0)
        self.grouper.assign_group(2, group0)

        group1 = self.grouper.create_or_get_group()
        self.grouper.assign_group(3, group1)
        self.grouper.assign_group(4, group1)

        # Merge groups
        self.grouper.merge_groups(group0, group1)
        self.assertEqual(len(self.grouper.groups[group0]), 4)
        self.assertFalse(group1 in self.grouper.groups)
        self.assertEqual(self.grouper.item_to_group[3], group0)
        self.assertEqual(self.grouper.item_to_group[4], group0)

    def test_full_groups(self):
        group0 = self.grouper.create_or_get_group()
        for i in range(30):
            self.grouper.assign_group(i, group0)

        self.assertFalse(self.grouper.group_is_full(group0))
        initial_group_len = len(self.grouper.groups[group0])

        group1 = self.grouper.create_or_get_group()
        for i in range(30, 33):
            self.grouper.assign_group(i, group1)

        self.grouper.merge_groups(group0, group1)
        # Assert no merge done
        self.assertEqual(len(self.grouper.groups[group0]), initial_group_len)
        self.assertTrue(group1 in self.grouper.groups)
        self.assertEqual(self.grouper.item_to_group[31], group1)
        self.assertEqual(self.grouper.item_to_group[32], group1)

    def test_replace_groups(self):
        group0 = self.grouper.create_or_get_group()
        for i in range(20):
            self.grouper.assign_group(i, group0)

        group1 = self.grouper.create_or_get_group()
        for i in range(20, 23):
            self.grouper.assign_group(i, group1)

        group2 = self.grouper.create_or_get_group()
        for i in range(23, 30):
            self.grouper.assign_group(i, group2)

        self.grouper.merge_groups(group1, group0)
        self.assertEqual(len(self.grouper.groups[group0]), 23)
        self.assertTrue(group1 in self.grouper.groups)
        self.assertFalse(group2 in self.grouper.groups)
        self.assertEqual(len(self.grouper.groups[group1]), 7)


def get_device():
    device = torch.device("cpu")
    return device


def test_get_scaled_distances():
    device = get_device()
    emb = torch.rand(4, 3)
    labels = torch.tensor([0, 1, 2, 2])
    distances = get_scaled_distances(emb, labels, device)
    assert torch.all(distances >= 0)
    assert torch.all(distances <= 1)


def test_batch_sorter_indices():
    device = get_device()
    inputs = torch.rand(32*16, 1000)
    labels = torch.rand(32*16, 1)
    masks = torch.rand(32*16, 100)
    embeddings = torch.rand(32*16, 32).to(device)

    i_out, l_out, m_out = sort_batches(inputs, labels, masks, embeddings, device)
    first_match_index = torch.all(inputs == i_out[0], dim=1).nonzero(as_tuple=True)[0][0]
    assert torch.all(labels[first_match_index] == l_out[0])
    assert torch.all(masks[first_match_index] == m_out[0])
