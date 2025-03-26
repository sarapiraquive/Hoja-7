from mdp._trial_interface import TrialInterface
import numpy as np

from policy_evaluation._linear import LinearSystemEvaluator
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator
from mdp._base import ClosedFormMDP

class ADPPolicyEvaluation(TrialBasedPolicyEvaluator):

    def __init__(
            self,
            trial_interface: TrialInterface,
            gamma: float,
            exploring_starts: bool,
            max_trial_length: int = np.inf,
            random_state: np.random.RandomState = None,
            precision_for_transition_probability_estimates=4,
            update_interval: int = 10
        ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            max_trial_length=max_trial_length,
            exploring_starts=exploring_starts,
            random_state=random_state
        )
        self.precision_for_transition_probability_estimates = precision_for_transition_probability_estimates
        self.update_interval = update_interval
    
    def process_trial_for_policy(self, df_trial, policy):
        """

        :param df_trial: dataframe with the trial (three columns with states, actions, and the rewards)
        :param policy: the policy that was used to create the trial
        :return: a dictionary with a report of the step
        """

        raise NotImplementedError