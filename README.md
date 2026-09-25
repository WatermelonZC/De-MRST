# De-MRST

This repository contains the PyTorch implementation, trained De-MRST checkpoints,
baselines, instances, and results for the paper *De-MRST: A Decentralized
Transformer for Routing and Scheduling in Marsupial Robotic Systems*.

## Install

Requirements: Python 3.9 or newer and PyTorch 2.7 or newer. Training requires
a CUDA GPU.

```bash
pip install -r requirements.txt
```

Gurobi is optional. To run that baseline, install `gurobipy` and provide a
valid Gurobi license.

## Predict

Run greedy decoding on one paper instance:

```bash
python scripts/predict.py \
  --instance benchmarks/instances/n20/01.json
```

The script selects `checkpoints/n20/best.pt` from the task count. It prints the
objective, the task–CR–WR plan, and execution metrics as JSON. Use
`--checkpoint PATH` for another compatible De-MRST, D-AM, or HetMRTA checkpoint.

For Sample-1280 decoding:

```bash
python scripts/predict.py \
  --instance benchmarks/instances/n20/01.json \
  --samples 1280 --device cuda --output prediction.json
```

`--batch-size` controls the number of sampled rollouts evaluated together
(default: 128).

## Train

The frozen `paper_n10`, `paper_n20`, `paper_n50`, and `paper_n100` protocols
select the paper's training settings. For N=20:

```bash
python scripts/train.py \
  --protocol-id paper_n20 \
  --method medp_1r \
  --model-seed 1103 \
  --device cuda \
  --output-dir runs/n20
```

The best validation checkpoint is written to `runs/n20/best.pt`. Replace
`paper_n20` and the output directory for another task count. The same trainer
accepts `d_am` and `hetmrta_mrs` for the decentralized learning baselines.
The centralized AM trainer is `scripts/train_c_am_formal.py`.

`checkpoints/manifest.json` records each supplied checkpoint's SHA-256,
training protocol, seed, and selected epoch.

## Baselines

The paper compares Gurobi, GA, ALNS, MURDOCH, Coalition-Auction, MinMin,
attention model (AM), and HetMRTA with De-MRST. Run a non-learning baseline:

```bash
python scripts/baselines.py \
  --instance benchmarks/instances/n20/01.json \
  --method d_min_min
```

Available methods are `d_min_min`, `d_murdoch`, `d_coalition_auction`,
`ga`, `alns`, `gurobi`, and `am`. The centralized methods use a 30 s cutoff
by default; change it with `--time-limit`. AM requires `--checkpoint PATH`.
Use `scripts/predict.py --checkpoint PATH` for trained D-AM and HetMRTA
policies.

For the Gurobi baseline:

```bash
python scripts/baselines.py \
  --instance benchmarks/instances/n20/01.json \
  --method gurobi --time-limit 30
```

To run one method over a paper instance directory and save each plan:

```bash
python scripts/evaluate.py \
  --instances benchmarks/instances/n20 \
  --method de_mrst \
  --output-dir runs/evaluation/n20_de_mrst
```

The runner writes `actions/*.json` and `summary.json`.

## Instances and results

- `benchmarks/instances/n10`, `n20`, `n50`, and `n100` contain the 20 JSON
  instances per task count used for the reported comparisons. Files are
  numbered `01.json` through `20.json`. The runner creates temporary XLSX
  input for the centralized baselines.
- `results/results.xlsx` contains the paper's reported records.

The 20 comparison instances were selected from the training protocol's fixed
validation seed stream. They are a shared comparison cohort, not an independent
held-out test set.

Run `python scripts/verify_release.py` to check protocol hashes, checkpoint
hashes, and strict checkpoint loading.

## Attribution

HetMRTA is a Marsupial-specific adaptation of
[marmotlab/HeteroMRTA](https://github.com/marmotlab/HeteroMRTA) (Apache 2.0).
The centralized attention model adapts code from
[wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)
under its [MIT license](ATTENTION_LICENSE.txt). The remaining methods are the
adaptations described in the paper. The repository code is released under the
[MIT license](LICENSE).
