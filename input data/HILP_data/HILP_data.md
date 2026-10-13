The data used in this directory are from the [HILP github repo](https://github.com/EECS-NTNU/hilp) ([`rodinia_profiled_data`](https://github.com/EECS-NTNU/hilp/tree/dbd3894fb8e9b5740cdc4ee6b92a6ed1fef83fdd/rodinia_profiled_data), commit `dbd3894`).

You can use `fetch_HILP_data.jl` to download them.

Run:
```
julia "input data/HILP_data/fetch_HILP_data.jl"
```
Please make sure to cite their work if you use the data as shown below:

```bibtex
@inproceedings{hilp,
  title        = {HILP: Accounting for Workload-Level Parallelism in System-on-Chip Design Space Exploration},
  booktitle    = {Proceedings of the International Symposium on High Performance Computer Architecture (HPCA)},
  author       = {Rogers, Joseph and Eeckhout, Lieven and Jahre, Magnus},
  year         = {2025}
}
```
