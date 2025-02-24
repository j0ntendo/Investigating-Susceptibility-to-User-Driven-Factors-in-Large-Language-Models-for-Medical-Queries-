import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

exclusion = ['No Exclusion', 'Physical Exam', 'Past History', 'History Taking', 'Demographic', 'Lab Diagnostic', 'Other']
medqa = {
    'Pharmacotherapy, Interventions and Management': {},
    'Diagnosis': {},
    'Health Maintenance, Prevention and Surveillance': {}
}

medbullet = {
    'Diagnosis': {},
    'Health Maintenance, Prevention and Surveillance': {},
    'Pharmacotherapy, Interventions and Management': {}
}

datasets = {'MedQA': medqa, 'MedBullet': medbullet}

colors = {
    'GPT-4o': '#5b9bd5',
    'Claude-3.5 Haiku': '#9e480e',
    'Claude-3.5 Sonnet': '#ed7d31',
    'Gemini-1.5 Pro': '#a5a5a5',
    'Gemini-1.5 Flash': '#ffc000',
    'Llama-3 8B': '#6a5acd',
    'Llama-3 Med42 8B': '#70ad47',
    'DeepSeek-R1 8B': '#d36b6b'
}

primary_models = ['Llama-3 Med42 8B', 'Llama-3 8B', 'DeepSeek-R1 8B']
secondary_models = ['GPT-4o', 'Claude-3.5 Haiku', 'Claude-3.5 Sonnet', 'Gemini-1.5 Pro', 'Gemini-1.5 Flash']

for dataset_name, data in datasets.items():
    for part, selected_models in zip(["OpenSourced LLMs", "Proprietary LLMs"], [primary_models, secondary_models]):
        fig, axes = plt.subplots(1, 3, figsize=(24, 10), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(exclusion), endpoint=False).tolist()
        angles += angles[:1]
        for ax, (category, models) in zip(axes, data.items()):
            ax.set_title(f"{category}", fontsize=18, fontweight='bold', pad=40)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(exclusion, fontsize=12, weight='bold')
            for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
                if angle in (0, np.pi):
                    label.set_horizontalalignment('center')
                elif 0 < angle < np.pi:
                    label.set_horizontalalignment('left')
                else:
                    label.set_horizontalalignment('right')
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=12, weight='bold')
            for model in selected_models:
                if model in models:
                    values = models[model] + models[model][:1]
                    color = colors.get(model, 'gray')
                    ax.plot(angles, values, linewidth=3, linestyle='solid', label=model, color=color)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=16, ncol=len(selected_models), title="AI Models", title_fontsize=14)
        plt.suptitle(f"{dataset_name} Ablation ({part})", fontsize=24, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{dataset_name}_category_{part}.png", dpi=300, bbox_inches='tight')
        plt.show()
