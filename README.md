# Conformal Prediction Intervals with Temporal Dependence

This is the code associated with "Conformal Prediction Intervals with Temporal Dependence".

To replicate the (`Load`) experiments:
1. Train the base models: `python -m utils.main_experiments`
2. run `main.ipynb` for results

For other datasets, please download the corresponding data and change the `__main__` section.

## Requirements
`numpy`, `torch`, `pandas`, `scipy`, `matplotlib`,  and `tqdm` 
(`jupyter` and `notebook` if you want to use the notebook) 
`env.yml` contains the full environment.