
class DatasetLoader():
    """Abstract class that serves to load datasets from different sources (local, ClearML, other tracker)
    """

    def get_dataset_folder(self, dataset_project, dataset_name):
        return NotImplementedError()


class LocalDatasetLoader(DatasetLoader):

    def get_dataset_folder(self, dataset_project, dataset_name):
        return f"data/{dataset_name}"
