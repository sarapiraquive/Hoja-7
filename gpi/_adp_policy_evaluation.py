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
        trial_length = len(df_trial)

        if self.workspace.v is None:
            self.workspace.replace_v({})

        updated_v = self.workspace.v.copy()  # Crear una copia de los valores actuales

        for t in range(trial_length - 1):
            state, action, reward = df_trial.iloc[t]
            next_state = df_trial.iloc[t + 1]["state"]

            if state not in updated_v:
                updated_v[state] = 0  # Inicializar si no está presente

            # Aplicar la actualización ADP
            updated_v[state] += (reward + self.gamma * updated_v.get(next_state, 0) - updated_v[
                state]) / self.update_interval

        # Reemplazar los valores de estado en workspace
        self.workspace.replace_v(updated_v)

        return {"trial_length": trial_length}