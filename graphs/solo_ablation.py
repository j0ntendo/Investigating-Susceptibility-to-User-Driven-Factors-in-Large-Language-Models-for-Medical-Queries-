import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def create_radar_chart(data, title, ax, colors):
    labels = ['No Exclusion', 'Demographic Data', 'History Taking', 
              'Past History', 'Physical Exam', 'Lab and Diagnostic', 'Other']
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  
    
    def add_to_radar(model_name, values, color):
        values += values[:1]  
        ax.plot(angles, values, linewidth=2, label=model_name, marker='o', markersize=4, color=color)  
    
    for model, values in data.items():
        add_to_radar(model, values, colors[model])  
    
    ax.set_theta_offset(np.pi / 2)  
    ax.set_theta_direction(-1)  
    ax.set_thetagrids(np.degrees(angles[:-1]), labels) 
    
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    ax.set_ylim(0, 100)
    ax.set_rlabel_position(180 / num_vars)
    ax.tick_params(colors='#222222') 
    ax.tick_params(axis='y', labelsize=8)  
    ax.grid(color='#AAAAAA') 
    ax.spines['polar'].set_color('#222222')  
    ax.set_facecolor('#FAFAFA')  
    
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=8, fontsize=10)
    # Data
    data1 = { "Llama 3 Med42 (8B)": [], "Llama 3 (8B)": [], "DeepSeek (8B)": [] } 
    data2 = { "GPT-4o": [], "Gemini 1.5 Pro": [], "Gemini 1.5 Flash": [], "Claude 3.5 Sonnet": [], "Claude 3.5 Haiku": [] }

    data3 = {
        "Llama 3 Med42 (8B)": [],
        "Llama 3 (8B)": [],
        "DeepSeek (8B)": []
    }
    data4 = {
        "GPT-4o": [],
        "Gemini 1.5 Pro": [],
        "Gemini 1.5 Flash": [],
        "Claude 3.5 Sonnet": [],
        "Claude 3.5 Haiku": []
    }

set1_colors = plt.get_cmap('Dark2').colors
set2_colors = plt.get_cmap('Set1').colors

combined_colors = set1_colors + set2_colors

all_models_medqa = list(data1.keys()) + list(data2.keys())
all_models_medbullets = list(data3.keys()) + list(data4.keys())

color_mapping_medqa = {model: combined_colors[i % len(combined_colors)] for i, model in enumerate(all_models_medqa)}
color_mapping_medbullets = {model: combined_colors[i % len(combined_colors)] for i, model in enumerate(all_models_medbullets)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), subplot_kw=dict(polar=True))
plt.subplots_adjust(wspace=0.2)
create_radar_chart(data1, "MedQA Ablation (OpenSourced LLMs)", ax1, {model: color_mapping_medqa[model] for model in data1.keys()})
create_radar_chart(data2, "MedQA Ablation (Proprietary LLMs)", ax2, {model: color_mapping_medqa[model] for model in data2.keys()})
plt.tight_layout()
plt.savefig("medqa_ablation.png", dpi=300, bbox_inches='tight')
plt.show()

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 6), subplot_kw=dict(polar=True))
plt.subplots_adjust(wspace=0.2)
create_radar_chart(data3, "MedBullets Ablation (OpenSourced LLMs)", ax3, {model: color_mapping_medbullets[model] for model in data3.keys()})
create_radar_chart(data4, "MedBullets Ablation (Proprietary LLMs)", ax4, {model: color_mapping_medbullets[model] for model in data4.keys()})
plt.tight_layout()
plt.savefig("medbullets_ablation.png", dpi=300, bbox_inches='tight')
plt.show()
