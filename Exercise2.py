import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from lake import LakeMDP
from gpi import _adp_policy_evaluation as adp
from gpi import _first_visit_monte_carlo_evaluator as fvmc
from gpi._base import GeneralPolicyIteration
from mdp._trial_interface import TrialInterface
from policy_evaluation._linear import LinearSystemEvaluator

# 1. Crear MDP determinista
lake_mdp = LakeMDP(probability_of_success=1.0)
trial_interface = TrialInterface(lake_mdp)

gamma = 0.9

# 2. Calcular valores reales usando LinearSystemEvaluator
linear_evaluator = LinearSystemEvaluator(lake_mdp, gamma)
linear_evaluator.reset(lake_mdp.optimal_policy)
true_values = linear_evaluator.v  # Obtener los valores reales de estado

# 3. Definir las configuraciones de los evaluadores
configs = [
    ("MC Exploring", fvmc.FirstVisitMonteCarloEvaluator(trial_interface, gamma, exploring_starts=True)),
    ("MC No Exploring", fvmc.FirstVisitMonteCarloEvaluator(trial_interface, gamma, exploring_starts=False)),
    ("ADP Exploring", adp.ADPPolicyEvaluation(trial_interface, gamma, exploring_starts=True, update_interval=10)),
    ("ADP No Exploring", adp.ADPPolicyEvaluation(trial_interface, gamma, exploring_starts=False, update_interval=10))
]

num_iterations = 10^5
errors = {name: [] for name, _ in configs}

# 4. Ejecutar cada técnica por separado
for name, evaluator in configs:
    gpi = GeneralPolicyIteration(gamma, [evaluator])  # Cada técnica tiene su propio GPI
    for _ in range(num_iterations):
        gpi.step()
        error_sum = 0
        for s in lake_mdp.states:
            v_est = gpi.components[0].workspace.v.get(s, 0)
            v_true = true_values.get(s, 0)
            error_sum += abs(v_est - v_true)
        errors[name].append(error_sum / len(lake_mdp.states))

# 5. Crear DataFrame con los errores
errors_df = DataFrame({
    'Iteración': range(1, num_iterations + 1),
    'Error MC Exploring Starts': errors['MC Exploring'],
    'Error MC No Exploring Starts': errors['MC No Exploring'],
    'Error ADP Exploring Starts': errors['ADP Exploring'],
    'Error ADP No Exploring Starts': errors['ADP No Exploring']
})

errors_df.to_csv('errores.csv', index=False)

# 6. Generar gráficos
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(errors_df['Iteración'], errors_df['Error MC Exploring Starts'], label='MC Exploring Starts', color='blue')
plt.title('Evolución Error MC (Exploring Starts)')
plt.xlabel('Iteración')
plt.ylabel('Error Promedio')
plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(errors_df['Iteración'], errors_df['Error MC No Exploring Starts'], label='MC No Exploring Starts', color='orange')
plt.title('Evolución Error MC (No Exploring Starts)')
plt.xlabel('Iteración')
plt.ylabel('Error Promedio')
plt.grid()
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(errors_df['Iteración'], errors_df['Error ADP Exploring Starts'], label='ADP Exploring Starts', color='green')
plt.title('Evolución Error ADP (Exploring Starts)')
plt.xlabel('Iteración')
plt.ylabel('Error Promedio')
plt.grid()
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(errors_df['Iteración'], errors_df['Error ADP No Exploring Starts'], label='ADP No Exploring Starts', color='red')
plt.title('Evolución Error ADP (No Exploring Starts)')
plt.xlabel('Iteración')
plt.ylabel('Error Promedio')
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig('errores_evolucion.png')
plt.show()

# 7. Imprimir valores finales y errores promedio
print("\nEstado | Valor Real | MC Exploring | MC No Exploring | ADP Exploring | ADP No Exploring")
for name, evaluator in configs:
    gpi = GeneralPolicyIteration(gamma, [evaluator])
    for _ in range(num_iterations):  # Re-ejecutar para obtener valores finales
        gpi.step()
    total_error = 0
    for s in lake_mdp.states:
        v_est = gpi.components[0].workspace.v.get(s, 0)
        v_true = true_values.get(s, 0)
        total_error += abs(v_est - v_true)
        if name == "MC Exploring":  # Imprimir solo una vez por estado
            print(f"{s} | {v_true:.4f} | {configs[0][1].workspace.v.get(s, 0):.4f} | "
                  f"{configs[1][1].workspace.v.get(s, 0):.4f} | {configs[2][1].workspace.v.get(s, 0):.4f} | "
                  f"{configs[3][1].workspace.v.get(s, 0):.4f}")
    print(f"Error absoluto promedio {name}: {total_error / len(lake_mdp.states):.4f}")