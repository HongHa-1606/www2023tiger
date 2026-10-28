# www2023tiger
Codes for WWW 2023 paper "TIGER: Temporal Interaction Graph Embedding with Restarts" [arXiv](https://arxiv.org/abs/2302.06057)

# Data

Please download data from the [project homepage of JODIE](https://snap.stanford.edu/jodie/) and pre-process them with the script provided by [TGN](https://github.com/twitter-research/tgn).

## Collecting Reddit interactions

The optional `scrapper.py` collects comment-author-to-submission interactions
through Reddit's read-only API. It writes a JODIE/TGN-style raw file and the
processed files consumed by TIGER (`data/ml_reddit.csv`,
`data/ml_reddit.npy`, and `data/ml_reddit_node.npy`). The collected data is
similar in schema and purpose to the JODIE Reddit dataset, but it is a new
dataset, not a copy of the official one. `state_label` is set to zero for
self-supervised link prediction; it is not a three-class label.

Set Reddit API credentials through environment variables (do not put them in
source code), install `praw`, then run from the repository root:

```powershell
$env:REDDIT_CLIENT_ID = "your-client-id"
$env:REDDIT_CLIENT_SECRET = "your-client-secret"
$env:REDDIT_USER_AGENT = "your-app-name/1.0"
python -m pip install praw
python scrapper.py --subreddit MachineLearning --target 25000
python train_self_supervised.py --data reddit --root . --gpu -1 --restart_prob 0 --restarter_type static
```

The scraper does not save usernames or comment text. Reddit limits listing
access to recent submissions; if fewer than the requested interactions are
available, it stops with an error instead of writing an undersized dataset.
Respect Reddit's API terms and rate limits.

### Running on Kaggle

Enable Internet in the Kaggle Notebook settings and add three Kaggle Notebook
Secrets named `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and
`REDDIT_USER_AGENT`. The scraper reads these directly and never prints their
values. In a Kaggle Notebook, clone the repo (or attach it as an input), change
to its directory, and run:

```python
!git clone https://github.com/HongHa-1606/www2023tiger.git /kaggle/working/www2023tiger
%cd /kaggle/working/www2023tiger
!pip install praw
!python scrapper.py --subreddit MachineLearning --target 25000 --output-dir /kaggle/working/www2023tiger/data
!python train_self_supervised.py --data reddit --root /kaggle/working/www2023tiger --gpu -1 --restart_prob 0 --restarter_type static
```

Kaggle's working directory is temporary; use the notebook's **Save Version**
or create an output dataset to preserve generated files. The Kaggle runtime
must also have the repository's model dependencies installed to train TIGER.

For the prepared Stage 3 Reddit dataset (`interactions_filtered.csv` and
`F.npy`), use [`run.ipynb`](./run.ipynb) to create chronological 70/15/15
event splits, `Test-new`/`Test-all` evaluation tables, and time-aware random
and historical negatives. Open/import the notebook from GitHub in Kaggle,
enable Internet so its first cell can clone this repository, and attach the
Stage 3 files as a Kaggle Input dataset (or set `DATA_DIR` in the first cell).
Its training cell saves checkpoints in `saved_models/`; use Kaggle **Save
Version** to preserve `/kaggle/working` outputs. The notebook's final cell
runs evaluation after training; set `RUN_TIGER_TRAINING = False` to prepare
artifacts without training. The repository's
existing runner still evaluates using its built-in quantile split and negative
sampler, so use the notebook's generated CSV files when evaluating the custom
Stage 4 protocol. After a successful training run, the notebook also replays
events online and writes Stage 6 metrics (`stage6_test_metrics.csv`), the
validation-selected baseline parameters, and a weekly MRR chart to the Stage 4
output directory. Random-negative AUC/AP is averaged across five seeds; ranking
metrics use the complete candidate set for the selected negative type.

# How to use

Temporal Link Prediction

```
python train_self_supervised.py --data [DATA] --msg_src [left/right] --upd_src [left/right] --restarter [seq/static] --restart_prob [0/0.001/...]
```
If you want to use mooc/lastfm datasets, please pass one more argument: `--dim 100`.

Temporal Link Prediction with multi-GPU
```
python train_self_supervised_ddp.py --gpu 0,1,2,3 [...and other arguments]
```

Node Classification
```
python train_supervised.py --code [CODE]
```
Here, [CODE] is the HASH code of a trained model with `train_self_supervised.py`.

# Cite

```
@inproceedings{zhang2023tiger,
  title={TIGER: Temporal Interaction Graph Embedding with Restarts},
  author={Zhang, Yao and Xiong, Yun and Liao, Yongxiang and Sun, Yiheng and Jin, Yucheng and Zheng, Xuehao and Zhu, Yangyong},
  booktitle={ACM Web Conference},
  year={2023}
}
```
