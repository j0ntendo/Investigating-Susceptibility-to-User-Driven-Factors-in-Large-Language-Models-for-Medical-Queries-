import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'

def create_side_by_side_radar_chart(data1, colors1, title1, data2, colors2, title2, filename):
    labels = ['No Exclusion', 'Demographic Data', 'History Taking', 
              'Past History', 'Physical Exam', 'Lab and Diagnostic', 'Other']
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(polar=True))
    plt.subplots_adjust(wspace=1.2)

    def add_to_radar(ax, data, colors, title):
        for model, values in data.items():
            values += values[:1]  
            ax.plot(angles, values, color=colors[model], linewidth=2, label=model, marker='o', markersize=6)  
        ax.set_theta_offset(np.pi / 2)  
        ax.set_theta_direction(-1)  
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=17) 
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
        ax.set_title(title, fontsize=20, fontweight='bold', y=1.15)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=8, fontsize=12)

    add_to_radar(axes[0], data1, colors1, title1)
    add_to_radar(axes[1], data2, colors2, title2)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    # Data for MedQA Ablation (OpenSourced vs. Proprietary LLMs)
    data1 = { "Llama-3 Med42 8B": [], "Llama-3 8B": [], "DeepSeek-R1 8B": [] } 
    data2 = { "GPT-4o": [], "Gemini-1.5 Pro": [], "Gemini-1.5 Flash": [], "Claude-3.5 Sonnet": [], "Claude-3.5 Haiku": [] }

    # Data for MedBullets Ablation (OpenSourced vs. Proprietary LLMs)
    data3 = { "Llama-3 Med42 8B": [], "Llama-3 8B": [], "DeepSeek-R1 8B": [] }
    data4 = { "GPT-4o": [], "Gemini-1.5 Pro": [], "Gemini-1.5 Flash": [], "Claude-3.5 Sonnet": [], "Claude-3.5 Haiku": [] }

colors1 = {"Llama-3 Med42 8B": "#70ad47", "Llama-3 8B": "#6a5acd", "DeepSeek-R1 8B": "#d36b6b"}
colors2 = {"GPT-4o": "#5b9bd5", "Gemini-1.5 Pro": "#a5a5a5", "Gemini-1.5 Flash": "#ffc000", 
           "Claude-3.5 Sonnet": "#ed7d31", "Claude-3.5 Haiku": "#9e480e"}
colors3 = {"Llama-3 Med42 8B": "#70ad47", "Llama-3 8B": "#6a5acd", "DeepSeek-R1 8B": "#d36b6b"}
colors4 = {"GPT-4o": "#5b9bd5", "Gemini-1.5 Pro": "#a5a5a5", "Gemini-1.5 Flash": "#ffc000", 
           "Claude-3.5 Sonnet": "#ed7d31", "Claude-3.5 Haiku": "#9e480e"}

create_side_by_side_radar_chart(data2, colors2, "MedQA Ablation (Proprietary LLMs)", 
                                data1, colors1, "MedQA Ablation (Opensource LLMs)", 
                                "medqa_ablation_combined.png")

create_side_by_side_radar_chart(data4, colors4, "MedBullets Ablation (Proprietary LLMs)", 
                                data3, colors3, "MedBullets Ablation (Opensource LLMs)", 
                                "medbullets_ablation_combined.png")