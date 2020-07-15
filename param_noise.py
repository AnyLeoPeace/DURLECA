'''The parameter-noise is from https://github.com/openai/baselines/'''

import numpy as np


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01, min_action_std = 0.01, std_adapt = None):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev
        self.std_adapt = std_adapt
        self.min_action_std = min_action_std

    def get_desired_action_stddev(self, step):
        if self.std_adapt is None:
            desired_action_stddev = self.desired_action_stddev
        else:
            desired_action_stddev = self.desired_action_stddev * self.std_adapt(step)
        
        return max(self.min_action_std, desired_action_stddev)

    def adapt(self, distance, step):
        desired_action_stddev = self.get_desired_action_stddev(step)
        print('current std',np.round(self.current_stddev, 8), 'distance', np.round(distance,8), 'desired std', np.round(desired_action_stddev,8))
        if distance > desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)
