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

# 3. Inicializar evaluadores y asignar workspace a través de GeneralPolicyIteration
fmvc_evaluator = fvmc.FirstVisitMonteCarloEvaluator(trial_interface, gamma, exploring_starts=True)
fmvc_evaluator_no_exploring = fvmc.FirstVisitMonteCarloEvaluator(trial_interface, gamma, exploring_starts=False)
adp_evaluator = adp.ADPPolicyEvaluation(trial_interface, gamma, exploring_starts=True, update_interval=10)
adp_evaluator_no_exploring = adp.ADPPolicyEvaluation(trial_interface, gamma, exploring_starts=False, update_interval=10)
components = [
    fmvc_evaluator,
    fmvc_evaluator_no_exploring,
    adp_evaluator,
    adp_evaluator_no_exploring
]

gpi = GeneralPolicyIteration(gamma, components)  # Crear GPI con los evaluadores

# 4. Ejecutar 105 iteraciones de GPI y recolectar errores promedio
num_iterations = 105
errors_mc = {'Exploring Starts': [], 'No Exploring Starts': []}
errors_adp = {'Exploring Starts': [], 'No Exploring Starts': []}

for _ in range(num_iterations):
    gpi.step()

    # Variables para sumar errores en cada iteración
    error_mc_exploring_sum = 0
    error_mc_no_exploring_sum = 0
    error_adp_exploring_sum = 0
    error_adp_no_exploring_sum = 0
    num_states = len(lake_mdp.states)

    # Calcular errores para cada estado y acumular
    for s in lake_mdp.states:
        v_mc_exploring = gpi.components[0].workspace.v.get(s, 0)
        v_mc_no_exploring = gpi.components[1].workspace.v.get(s, 0)
        v_adp_exploring = gpi.components[2].workspace.v.get(s, 0)
        v_adp_no_exploring = gpi.components[3].workspace.v.get(s, 0)
        v_true = true_values.get(s, 0)

        error_mc_exploring_sum += abs(v_mc_exploring - v_true)
        error_mc_no_exploring_sum += abs(v_mc_no_exploring - v_true)
        error_adp_exploring_sum += abs(v_adp_exploring - v_true)
        error_adp_no_exploring_sum += abs(v_adp_no_exploring - v_true)

    # Calcular promedio y añadir a las listas
    errors_mc['Exploring Starts'].append(error_mc_exploring_sum / num_states)
    errors_mc['No Exploring Starts'].append(error_mc_no_exploring_sum / num_states)
    errors_adp['Exploring Starts'].append(error_adp_exploring_sum / num_states)
    errors_adp['No Exploring Starts'].append(error_adp_no_exploring_sum / num_states)

# 5. Comparar valores aprendidos con valores reales y calcular error absoluto promedio
print("Estado | Valor Real | MC Exploring | MC No Exploring | ADP Exploring | ADP No Exploring")
total_error_mc_exploring = 0
total_error_mc_no_exploring = 0
total_error_adp_exploring = 0
total_error_adp_no_exploring = 0

for s in lake_mdp.states:
    v_mc_exploring = gpi.components[0].workspace.v.get(s, 0)
    v_mc_no_exploring = gpi.components[1].workspace.v.get(s, 0)
    v_adp_exploring = gpi.components[2].workspace.v.get(s, 0)
    v_adp_no_exploring = gpi.components[3].workspace.v.get(s, 0)
    v_true = true_values.get(s, 0)

    error_mc_exploring = abs(v_mc_exploring - v_true)
    error_mc_no_exploring = abs(v_mc_no_exploring - v_true)
    error_adp_exploring = abs(v_adp_exploring - v_true)
    error_adp_no_exploring = abs(v_adp_no_exploring - v_true)

    total_error_mc_exploring += error_mc_exploring
    total_error_mc_no_exploring += error_mc_no_exploring
    total_error_adp_exploring += error_adp_exploring
    total_error_adp_no_exploring += error_adp_no_exploring

    print(
        f"{s} | {v_true:.4f} | {v_mc_exploring:.4f} | {v_mc_no_exploring:.4f} | {v_adp_exploring:.4f} | {v_adp_no_exploring:.4f}")

avg_error_mc_exploring = total_error_mc_exploring / len(lake_mdp.states)
avg_error_mc_no_exploring = total_error_mc_no_exploring / len(lake_mdp.states)
avg_error_adp_exploring = total_error_adp_exploring / len(lake_mdp.states)
avg_error_adp_no_exploring = total_error_adp_no_exploring / len(lake_mdp.states)

print(f"\nError absoluto promedio MC Exploring Starts: {avg_error_mc_exploring:.4f}")
print(f"Error absoluto promedio MC No Exploring Starts: {avg_error_mc_no_exploring:.4f}")
print(f"Error absoluto promedio ADP Exploring Starts: {avg_error_adp_exploring:.4f}")
print(f"Error absoluto promedio ADP No Exploring Starts: {avg_error_adp_no_exploring:.4f}")

# 6. Verificar si coinciden después de 105 iteraciones
tolerance = 1e-3  # Pequeño margen de error numérico
mc_exploring_matches = all(
    abs(gpi.components[0].workspace.v.get(s, 0) - true_values.get(s, 0)) < tolerance for s in lake_mdp.states)
mc_no_exploring_matches = all(
    abs(gpi.components[1].workspace.v.get(s, 0) - true_values.get(s, 0)) < tolerance for s in lake_mdp.states)
adp_exploring_matches = all(
    abs(gpi.components[2].workspace.v.get(s, 0) - true_values.get(s, 0)) < tolerance for s in lake_mdp.states)
adp_no_exploring_matches = all(
    abs(gpi.components[3].workspace.v.get(s, 0) - true_values.get(s, 0)) < tolerance for s in lake_mdp.states)

print("\n¿MC Exploring coincide con los valores reales después de 105 iteraciones?:", mc_exploring_matches)
print("¿MC No Exploring coincide con los valores reales después de 105 iteraciones?:", mc_no_exploring_matches)
print("¿ADP Exploring coincide con los valores reales después de 105 iteraciones?:", adp_exploring_matches)
print("¿ADP No Exploring coincide con los valores reales después de 105 iteraciones?:", adp_no_exploring_matches)

# 7. Guardar errores en DataFrame y exportar a CSV
errors_df = DataFrame({
    'Iteración': range(1, num_iterations + 1),
    'Error MC Exploring Starts': errors_mc['Exploring Starts'],
    'Error MC No Exploring Starts': errors_mc['No Exploring Starts'],
    'Error ADP Exploring Starts': errors_adp['Exploring Starts'],
    'Error ADP No Exploring Starts': errors_adp['No Exploring Starts']
})

errors_df.to_csv('errores.csv', index=False)

# 8. Generar gráficos
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(errors_df['Iteración'], errors_df['Error MC Exploring Starts'], label='MC Exploring Starts')
plt.title('Evolución Error MC (Exploring Starts)')
plt.xlabel('Iteración')
plt.ylabel('Error Promedio')
plt.grid()
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(errors_df['Iteración'], errors_df['Error MC No Exploring Starts'], label='MC No Exploring Starts',
         color='orange')
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
plt.plot(errors_df['Iteración'], errors_df['Error ADP No Exploring Starts'], label='ADP No Exploring Starts',
         color='red')
plt.title('Evolución Error ADP (No Exploring Starts)')
plt.xlabel('Iteración')
plt.ylabel('Error Promedio')
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig('errores_evolucion.png')
plt.show()