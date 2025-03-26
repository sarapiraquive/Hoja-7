from _base import GeneralPolicyIterationComponent
from mdp import ClosedFormMDP
from mdp._trial_interface import TrialInterface
import numpy as np
import pandas as pd
from abc import abstractmethod


class TrialBasedPolicyEvaluator(GeneralPolicyIterationComponent):

    def __init__(
            self,
            trial_interface: TrialInterface,
            gamma: float,
            exploring_starts: bool,
            max_trial_length: int = np.inf,
            random_state: np.random.RandomState = None
        ):
        super().__init__()
        self.trial_interface = trial_interface
        self.gamma = gamma
        self.max_trial_length = max_trial_length
        self.exploring_starts = exploring_starts
        self.random_state = random_state
    
    def step(self):
        """
            creates and processes a trial to update state-values and q-values
        """
        pass
    
    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError