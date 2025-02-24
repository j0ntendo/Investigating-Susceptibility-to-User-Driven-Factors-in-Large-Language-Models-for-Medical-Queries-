import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['hatch.linewidth'] = 5
class CustomHatchHandler(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        patch = mpatches.Rectangle([xdescent, ydescent], width, height, facecolor=orig_handle.get_facecolor(), hatch=orig_handle.get_hatch(), edgecolor=orig_handle.get_edgecolor(), linewidth=orig_handle.get_linewidth(), transform=trans)
        patch.set_hatch(r'\\')
        return [patch]

models = ['GPT-4o','Claude-3.5 Sonnet','Claude-3.5 Haiku', 'Gemini-1.5 Pro', 
          'Gemini-1.5 Flash', 'Llama-3 8B', 'Llama-3 Med42 8B', 'DeepSeek-R1 8B']

variations = ["Original prompt",
"Hedged Tone/Novice physician/Medical expert AI",
"Hedged Tone/Novice physician/Medical assistant AI",
"Hedged Tone/Expert physician/Medical expert AI",
"Hedged Tone/Expert physician/Medical assistant AI",
"Definitive Tone/Novice physician/Medical expert AI",
"Definitive Tone/Novice physician/Medical assistant AI",
"Definitive Tone/Expert physician/Medical expert AI",
"Definitive Tone/Expert physician/Medical assistant AI"]
Pharmacotherapy_Interventions_and_Management = np.array([], dtype=object)
Applying_Foundational_Science_Concepts = np.array([], dtype=object)
Diagnosis = np.array([], dtype=object)
HealthMaintenance_Prevention_and_Surveillance = np.array([], dtype=object)

Pharmacotherapy_Interventions_and_Management_counts = {
    'GPT-4o': (0, 0),
    'Claude-3.5 Sonnet': (0, 0),
    'Claude-3.5 Haiku': (0, 0),
    'Gemini-1.5 Pro': (0, 0),
    'Gemini-1.5 Flash': (0, 0),
    'Llama-3 8B': (0, 0),
    'Llama-3 Med42 8B': (0, 0),
    'DeepSeek-R1 8B': (0, 0)
}
Applying_Foundational_Science_Concepts_counts = {
    'GPT-4o': (0, 0),
    'Claude-3.5 Sonnet': (0, 0),
    'Claude-3.5 Haiku': (0, 0),
    'Gemini-1.5 Pro': (0, 0),
    'Gemini-1.5 Flash': (0, 0),
    'Llama-3 8B': (0, 0),
    'Llama-3 Med42 8B': (0, 0),
    'DeepSeek-R1 8B': (0, 0)
}
Diagnosis_counts = {
    'GPT-4o': (0, 0),
    'Claude-3.5 Sonnet': (0, 0),
    'Claude-3.5 Haiku': (0, 0),
    'Gemini-1.5 Pro': (0, 0),
    'Gemini-1.5 Flash': (0, 0),
    'Llama-3 8B': (0, 0),
    'Llama-3 Med42 8B': (0, 0),
    'DeepSeek-R1 8B': (0, 0)
}
HealthMaintenance_Prevention_and_Surveillance_counts = {
    'GPT-4o': (0, 0),
    'Claude-3.5 Sonnet': (0, 0),
    'Claude-3.5 Haiku': (0, 0),
    'Gemini-1.5 Pro': (0, 0),
    'Gemini-1.5 Flash': (0, 0),
    'Llama-3 8B': (0, 0),
    'Llama-3 Med42 8B': (0, 0),
    'DeepSeek-R1 8B': (0, 0)
}



def paired_permutation_test(baseline, perturbed, n_permutations=10000):
    observed_diff = np.mean(perturbed) - np.mean(baseline)
    combined = np.concatenate((baseline, perturbed))
    n = len(baseline)
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_baseline = combined[:n]
        perm_perturbed = combined[n:]
        perm_diff = np.mean(perm_perturbed) - np.mean(perm_baseline)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return count/n_permutations

def plot_data(data, ci_lower, ci_upper, p_vals, title, save_path):
    x = np.arange(len(variations))
    width = 0.6
    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharey=False)
    axes = axes.flatten()
    styles = [{'color': '#333333', 'hatch': ''},
              {'color': '#7698d4', 'hatch': ''},
              {'color': '#7698d4', 'hatch': r'\\'},
              {'color': '#2f5597', 'hatch': ''},
              {'color': '#2f5597', 'hatch': r'\\'},
              {'color': '#ff6565', 'hatch': ''},
              {'color': '#ff6565', 'hatch': r'\\'},
              {'color': '#c00000', 'hatch': ''},
              {'color': '#c00000', 'hatch': r'\\'}]
    for i, model in enumerate(models):
        ax = axes[i]
        for j in range(len(variations)):
            if styles[j]['hatch']:
                facecolor = 'white'
                edge_color = styles[j]['color']
                lw = 2
            else:
                facecolor = styles[j]['color']
                edge_color = 'none'
                lw = 0
            ax.bar(x[j], data[i, j], width, facecolor=facecolor, hatch=styles[j]['hatch'], edgecolor=edge_color, linewidth=lw, alpha=0.8)
            ax.errorbar(x[j], data[i, j], yerr=[[ci_lower[i, j]], [ci_upper[i, j]]], fmt="none", capsize=5, color="black", elinewidth=1.5)
            p_val = p_vals[i, j]
            if p_val < 0.001:
                star = '***'
            elif p_val < 0.01:
                star = '**'
            elif p_val < 0.05:
                star = '*'
            else:
                star = ''
            if star:
                ax.text(x[j], data[i, j] + ci_upper[i, j] + 2, star, ha='center', va='bottom', fontsize=12, color='black')
        ax.set_title(model, fontsize=16, fontweight='bold', pad=10)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xticks([])
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.yaxis.set_tick_params(labelleft=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        if i in [0, 1]:
            legend_elements = [mpatches.Patch(facecolor='black', label='Expert'), mpatches.Patch(facecolor='none', hatch='\\', edgecolor='black', label='Assistant')]
            ax.legend(handles=legend_elements, title='AI persona', loc='center right', bbox_to_anchor=(0.95, 0.4), frameon=True, fontsize=12, edgecolor='black', handler_map={mpatches.Patch: CustomHatchHandler()})
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


categories = {
    "Applying Foundational Science Concepts": (Applying_Foundational_Science_Concepts, Applying_Foundational_Science_Concepts_counts, "medqa_applying_foundational_science.png"),
    "Diagnosis": (Diagnosis, Diagnosis_counts, "medqa_diagnosis.png"),
    "Pharmacotherapy, Interventions and Management": (Pharmacotherapy_Interventions_and_Management, Pharmacotherapy_Interventions_and_Management_counts, "medqa_pharmacotherapy_interventions_and_management.png"),
    "Health Maintenance, Prevention and Surveillance": (HealthMaintenance_Prevention_and_Surveillance, HealthMaintenance_Prevention_and_Surveillance_counts, "medqa_health_maintenance_prevention_and_surveillance.png")
}

for cat_name, (cat_array, cat_counts, filename) in categories.items():
    data = np.array([[t/(t+f)*100 for t, f in row] for row in cat_array])
    ci_lower = np.zeros((len(models), len(variations)))
    ci_upper = np.zeros((len(models), len(variations)))
    p_values = np.zeros((len(models), len(variations)))
    for i, model in enumerate(models):
        baseline_true, baseline_false = cat_counts[model]
        baseline_total = baseline_true + baseline_false
        for j in range(len(variations)):
            perturbed_true, perturbed_false = cat_array[i][j]
            perturbed_total = perturbed_true + perturbed_false
            obs_perturbed = perturbed_true/perturbed_total*100
            baseline_bs = [np.random.binomial(n=baseline_total, p=baseline_true/baseline_total)/baseline_total*100 for _ in range(1000)]
            perturbed_bs = [np.random.binomial(n=perturbed_total, p=perturbed_true/perturbed_total)/perturbed_total*100 for _ in range(1000)]
            boot_ci = np.percentile(perturbed_bs, [2.5, 97.5])
            ci_lower[i, j] = obs_perturbed - boot_ci[0]
            ci_upper[i, j] = boot_ci[1] - obs_perturbed
            p_values[i, j] = paired_permutation_test(baseline_bs, perturbed_bs)
    adjusted_p_values = multipletests(p_values.flatten(), method="fdr_bh")[1].reshape(p_values.shape)
    plot_data(data, ci_lower, ci_upper, adjusted_p_values, "LLM Performance Across Perturbations (MedQA)\n" + cat_name, filename)