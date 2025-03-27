import numpy as np

from mdp._trial_interface import TrialInterface
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator


class FirstVisitMonteCarloEvaluator(TrialBasedPolicyEvaluator):

    def __init__(
            self,
            trial_interface: TrialInterface,
            gamma: float,
            exploring_starts: bool,
            max_trial_length: int = np.inf,
            random_state: np.random.RandomState = None
        ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            exploring_starts=exploring_starts,
            max_trial_length=max_trial_length,
            random_state=random_state
        )

    def process_trial_for_policy(self, df_trial, policy):
        """

        :param df_trial: dataframe with the trial (three columns with states, actions, and the rewards)
        :param policy: the policy that was used to create the trial
        :return: a dictionary with a report of the step
        """
        G = 0
        visited_states = set()
        trial_length = len(df_trial)
        state_counts = {}  # Contador de visitas por estado

        # Inicializar diccionario temporal de valores si es necesario
        if self.workspace.v is None:
            self.workspace.replace_v({})

        updated_v = self.workspace.v.copy()  # Crear una copia para modificar

        # Iterar sobre el ensayo en orden inverso
        for t in reversed(range(trial_length)):
            state, _, reward = df_trial.iloc[t]
            G = self.gamma * G + reward  # Calcular retorno acumulado

            # Solo actualizar el valor del estado en la primera visita
            if state not in visited_states:
                visited_states.add(state)
                if state not in updated_v:
                    updated_v[state] = 0  # Inicializar si el estado no está presente
                if state not in state_counts:
                    state_counts[state] = 1  # Inicializar correctamente en 1
                else:
                    state_counts[state] += 1

                # Aplicar actualización Monte Carlo con promedio incremental
                updated_v[state] += (G - updated_v[state]) / state_counts[state]


        print("📌 Valores V actualizados:", updated_v)

        # Reemplazar los valores de estado en workspace
        self.workspace.replace_v(updated_v)

        return {"trial_length": trial_length}