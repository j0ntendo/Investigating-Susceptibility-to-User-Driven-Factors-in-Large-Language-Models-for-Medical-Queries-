import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['hatch.linewidth'] = 3

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
medbullet = np.array([
    # GPT-4o
    [],
    # Claude-3.5 Sonnet
    [],
    # Claude-3.5 Haiku
    [],
    # Gemini-1.5 Pro
    [],
    # Gemini-1.5 Flash
    [],
    # Llama-3 8B
    [],
    # Llama-3 Med42 8B
    [],
    # DeepSeek-R1 8B
    []
], dtype=object)

baseline_counts = {
    'GPT-4o': (0, 0),
    'Claude-3.5 Haiku': (0, 0),
    'Claude-3.5 Sonnet': (0, 0),
    'Gemini-1.5 Pro': (0, 0),
    'Gemini-1.5 Flash': (0, 0),
    'Llama-3 8B': (0, 0),
    'Llama-3 Med42 8B': (0, 0),
    'DeepSeek-R1 8B': (0, 0)
}

data = np.array([[t/(t+f)*100 for t, f in model_data] for model_data in medbullet])

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

p_values = np.zeros((len(models), len(variations)))
ci_lower = np.zeros((len(models), len(variations)))
ci_upper = np.zeros((len(models), len(variations)))

for i, model in enumerate(models):
    baseline_true, baseline_false = baseline_counts[model]
    baseline_total = baseline_true + baseline_false
    for j in range(len(variations)):
        perturbed_true, perturbed_false = medbullet[i][j]
        perturbed_total = perturbed_true + perturbed_false
        obs_perturbed = perturbed_true/perturbed_total*100
        baseline_bs = [np.random.binomial(n=baseline_total, p=baseline_true/baseline_total)/baseline_total*100 for _ in range(1000)]
        perturbed_bs = [np.random.binomial(n=perturbed_total, p=perturbed_true/perturbed_total)/perturbed_total*100 for _ in range(1000)]
        boot_ci = np.percentile(perturbed_bs, [2.5, 97.5])
        ci_lower[i, j] = obs_perturbed - boot_ci[0]
        ci_upper[i, j] = boot_ci[1] - obs_perturbed
        p_values[i, j] = paired_permutation_test(baseline_bs, perturbed_bs)

adjusted_p_values = multipletests(p_values.flatten(), method="fdr_bh")[1].reshape(p_values.shape)

plt.rcParams['hatch.linewidth'] = 5

def plot_data(data, ci_lower, ci_upper, p_vals, title, save_path):
    x = np.arange(len(variations))
    width = 0.6
    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharey=False)  # Set sharey=False
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
            legend_elements = [mpatches.Patch(facecolor='black', label='Expert'),
                               mpatches.Patch(facecolor='none', hatch='//', edgecolor='black', label='Assistant')]
            ax.legend(handles=legend_elements, title='AI persona', loc='center right', bbox_to_anchor=(0.95, 0.4), frameon=True, fontsize=12, edgecolor='black')  # Encased legend in a box

    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


#pval
excel_data = []
for i, model in enumerate(models):
    for j in range(len(variations)):
        accuracy = data[i, j]
        p_val = adjusted_p_values[i, j]
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        excel_data.append([model, variations[j], f"{accuracy:.2f}%", f"{p_val:.5f}", sig])

excel_df = pd.DataFrame(excel_data, columns=["Model", "Variation", "Accuracy (%)", "P-value", "Significance"])
excel_filename = "medbullet_permutation_results.xlsx"
excel_df.to_excel(excel_filename, index=False, sheet_name="Permutation Test Results")

plot_data(data, ci_lower, ci_upper, adjusted_p_values, "LLM Performance Across Perturbations (Medbullet)", "newmedbullet_permuation_graph.png")
print(f"Excel file '{excel_filename}' created successfully!")