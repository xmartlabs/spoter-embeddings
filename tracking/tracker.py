
class Tracker:

    def __init__(self, project_name, experiment_name):
        super().__init__()

    def execute_remotely(self, queue_name):
        pass

    def track_config(self, configs):
        # Used to track configuration parameters of an experiment run
        pass

    def track_artifacts(self, filepath):
        # Used to track artifacts like model weights
        pass

    def log_scalar_metric(self, metric, series, iteration, value):
        pass

    def log_chart(self, title, series, iteration, figure):
        pass

    def finish_run(self):
        pass

    def get_callback(self):
        pass
