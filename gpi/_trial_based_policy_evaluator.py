from gpi._base import GeneralPolicyIterationComponent
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

        def optimal_policy(s):
            policy_map = {
                (0, 0): "r", (0, 1): "r", (0, 2): "d", (0, 3): "d",
                (1, 0): "d", (1, 1): "r", (1, 2): "d", (1, 3): "l",
                (2, 0): "r", (2, 1): "r", (2, 2): "d", (2, 3): "l",
                (3, 0): "u", (3, 1): "r", (3, 2): "r", (3, 3): None  # Meta
            }
            return policy_map.get(s, "r")  # Si no está definido, mover a la derecha por defecto

        trial = self.trial_interface.exec_policy(optimal_policy)
        return self.process_trial_for_policy(trial, optimal_policy)
    
    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError