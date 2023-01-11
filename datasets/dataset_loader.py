
import os


class DatasetLoader():
    """Abstract class that serves to load datasets from different sources (local, ClearML, other tracker)
    """

    def get_dataset_folder(self, dataset_project, dataset_name):
        return NotImplementedError()


class LocalDatasetLoader(DatasetLoader):

    def get_dataset_folder(self, dataset_project, dataset_name):
        base_folder = os.environ.get("BASE_DATA_FOLDER", "data")
        return os.path.join(base_folder, dataset_name)
