from ._base import PolicyEvaluator
import numpy as np
from mdp._mdp_utils import get_closed_form_of_mdp


class LinearSystemEvaluator(PolicyEvaluator):

    def __init__(self, mdp, gamma):
        super().__init__(gamma)
        self.mdp = mdp
        self.states, self.probs, self.rewards = get_closed_form_of_mdp(mdp)
        self.n = len(self.states)
        self._v_values = {s: 0 for s in self.states}  # Inicializar valores de estado en 0

    def _after_reset(self):
        """
            Update q-values
        """
        gamma_adj = min(self.gamma, 0.9999)

        # Crear la matriz A y el vector y
        A = np.eye(self.n)  # Matriz identidad de tamaño n
        y = np.zeros(self.n)  # Vector de términos independientes

        # Establecer las recompensas para todos los estados
        for i, s in enumerate(self.states):
            y[i] = self.rewards[i]
            actions = self.mdp.get_actions_in_state(s)
            if not actions:  # Si es un estado terminal, mantener el valor actual
                continue

            action = self.policy(s)
            if action is None:
                print(f"Warning: Undefined policy for state {s}.")
                continue

            # Actualizar la matriz A para estados no terminales
            for j, s_prime in enumerate(self.states):
                if s_prime in self.probs[s][action]:
                    A[i, j] -= gamma_adj * self.probs[s][action][s_prime]

        try:
            v_values = np.linalg.solve(A, y)
            self._v_values = {s: v_values[i] for i, s in enumerate(self.states)}
        except np.linalg.LinAlgError as e:
            print(f"Error solving: {e}")

    @property
    def provides_state_values(self):
        return True

    @property
    def v(self):
        return self._v_values

    @property
    def q(self):
        """
            calcula Q(s,a) a partir de v(s)
        """
        q_values = {}
        for s in self.states:
            actions = self.mdp.get_actions_in_state(s)
            if not actions:  # Si es un estado terminal, no tendrá valores Q
                continue
            q_values[s] = {}
            for a in actions:
                if s in self.probs and a in self.probs[s]:
                    q_values[s][a] = self.rewards[self.states.index(s)] + self.gamma * sum(
                        self.probs[s][a].get(s_prime, 0) * self._v_values.get(s_prime, 0) for s_prime in self.states
                    )
        return q_values

