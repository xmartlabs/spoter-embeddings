from collections import deque
import numpy as np


class BatchingScheduler():
    """ This class acts as scheduler for the batching algorithm
    """

    def __init__(self, decay_factor=0.8, min_threshold=0.2, triplets_threshold=10, cooldown=10) -> None:
        # internal vars
        self._step_count = 0
        self._dist_threshold = 0.5
        self._last_used_triplets = deque([], 5)
        self._scaling_same_label_factor = 1
        self._last_update_step = -10

        # Parameters
        self.decay_factor = decay_factor
        self.min_threshold = min_threshold
        self.triplets_threshold = triplets_threshold
        self.cooldown = cooldown

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self, used_triplets):
        self._step_count += 1
        self._last_used_triplets.append(used_triplets)
        if (np.mean(self._last_used_triplets) < self.triplets_threshold and
                self._last_update_step + self.cooldown <= self._step_count):
            if self._dist_threshold > self.min_threshold:
                print(f"Updating dist_threshold at {self._step_count} ({np.mean(self._last_used_triplets)})")
                self.update_dist_threshold()
            if self._scaling_same_label_factor > 0.6:
                print(f"Updating scale factor at {self._step_count} ({np.mean(self._last_used_triplets)})")
                self.update_scale_factor()
            self._last_update_step = self._step_count

    def update_scale_factor(self):
        self._scaling_same_label_factor = max(self._scaling_same_label_factor * 0.9, 0.6)
        print(f"Updating scaling factor to {self._scaling_same_label_factor}")

    def update_dist_threshold(self):
        self._dist_threshold = max(self.min_threshold, self._dist_threshold * self.decay_factor)
        print(f"Updated dist_threshold to {self._dist_threshold}")

    def get_dist_threshold(self) -> float:
        return self._dist_threshold

    def get_scaling_same_label_factor(self) -> float:
        return self._scaling_same_label_factor
