from clearml import Dataset
from .dataset_loader import DatasetLoader

class ClearMLDatasetLoader(DatasetLoader):

    def get_dataset_folder(self, dataset_project, dataset_name):
        return Dataset.get(dataset_project=dataset_project, dataset_name=dataset_name).get_local_copy()
