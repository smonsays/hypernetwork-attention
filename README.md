# Attention as a Hypernetwork

Official code to reproduce experiments in [Attention as a Hypernetwork](https://arxiv.org/abs/2406.05816).

## Installation

Install jax according to the [instructions for your platform](https://jax.readthedocs.io/en/latest/installation.html) after which you can install the remaining dependencies with:
```
pip install -r requirements.txt
```

## Structure

All experiments have a corresponding sweep file in `sweeps/` and can be run using
```bash
`wandb sweep /sweeps/[name].yaml`
```
Default hyperparameters for all methods and experiments can be found in `configs/`.
If you'd like to directly run a specific experiment for a single seed you can use:

```bash
python run.py --config 'configs/[experiment].py:[method]'
```

where `experiment` $\in$ [`logic`, `raven`, `wiki`] and `method` $\in$ [`softmax_attention`, `linear_attention`, `linear_hypatt`].

## Citation

If you use this code in your research, please cite the paper:

```
@article{2024hyperatt,
  title={Attention as a Hypernetwork}, 
  author={Simon Schug and Seijin Kobayashi and Yassir Akram and João Sacramento and Razvan Pascanu},
  year={2024},
  url = {https://arxiv.org/abs/2406.05816},
}
```
