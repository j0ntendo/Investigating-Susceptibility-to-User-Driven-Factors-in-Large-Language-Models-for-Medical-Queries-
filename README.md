# Investigating Susceptibility to User-Driven Factors in Large Language Models for Medical Queries

This repository contains code for the paper "Investigating Susceptibility to User-Driven Factors in Large Language Models for Medical Queries"

Installation

pip install -r requirements.txt

Use your own MEDQA and MedBullets datasets, API keys, and a suitable GPU for Ollama.

Experiments

1. Perturbation Test

python run_medqa_8var.py
python run_medbullet_8var.py

2. Ablation Test
	1.	Create six JSON files that exclude specific types of information.
	2.	Run these scripts:

python run_medqa_ablation.py
python run_medbullet_ablation.py

Generating Graphs
Fill in the required information in the Python scripts that handle graph generation.