## Monte Carlo evaluation

Evaluates the nominal and robust designs from the optimization models under uncertainty in total carbon parameters using Monte Carlo.

### Run

```
pip install -r evaluation/requirements.txt
python evaluation/run_mc_evaluation.py
```

Defaults (workload, Nwork, designs, parameter values) are in [mc_evaluation/config.py](mc_evaluation/config.py). Any of them can be overridden from the command line, e.g.

```
python evaluation/run_mc_evaluation.py --workload bfs --nwork 1e4 --nominal c3_g46_f765 --robust c2_g10_f765
python evaluation/run_mc_evaluation.py --distributions Normal Uniform --verify
```

Run `python evaluation/run_mc_evaluation.py --help` for all options. Results are saved to `evaluation/results/mc_comparison_all_distributions_*.csv`.

### Files

| File | Contents |
|---|---|
| `run_mc_evaluation.py` | run this file |
| `mc_evaluation/config.py` | inputs and constraints |
| `mc_evaluation/design_space.py` | loads nominal data |
| `mc_evaluation/distributions.py` | samplers and the uncertainty scenario for each distribution family |
| `mc_evaluation/monte_carlo.py` | propagates the parameter samples |
| `mc_evaluation/analysis.py` | nominal vs. robust statistics and constraint violations |

### Modeling notes

- All designs are evaluated on the same parameter samples, so differences between the nominal and robust designs come from the designs, not from the random draws.
- Each distribution is defined by a mean and standard deviation (σ = the full spread of each parameter's range) and a lower bound. Samples below the bound are redrawn. 
