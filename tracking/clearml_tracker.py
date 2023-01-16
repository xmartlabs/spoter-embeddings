from clearml import Task, Logger
from .tracker import Tracker


class ClearMLTracker(Tracker):

    def __init__(self, project_name=None, experiment_name=None):
        self.task = Task.current_task() or Task.init(project_name=project_name, task_name=experiment_name)

    def execute_remotely(self, queue_name):
        self.task.execute_remotely(queue_name=queue_name)

    def log_scalar_metric(self, metric, series, iteration, value):
        Logger.current_logger().report_scalar(metric, series, iteration=iteration, value=value)

    def log_chart(self, title, series, iteration, figure):
        Logger.current_logger().report_plotly(title=title, series=series, iteration=iteration, figure=figure)

    def finish_run(self):
        self.task.mark_completed()
        self.task.close()
