from ._base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np


class StandardTrialInterfaceBasedPolicyImprover(GeneralPolicyIterationComponent):

    def __init__(self, trial_interface: TrialInterface, random_state: np.random.RandomState):
        super().__init__()
        self.trial_interface = trial_interface
        self.random_state = random_state
    
    def step(self):
        """
            Improves the current policy based on current q-values
        """
        pass