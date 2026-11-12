# Results: TCS.NS next-day direction (2015-01-01 to 2025-12-31)

> **SYNTHETIC DATA. These numbers only prove the pipeline runs; they say nothing about real markets.**

Class balance (share of UP days): train 51.2%, val 47.1%, test 46.9%. Majority-class accuracy on test = 53.1%.

## Model comparison (test set, mean over seeds)

| model                  |   accuracy_mean |   accuracy_std |   precision_mean |   precision_std |   recall_mean |   recall_std |   f1_mean |   f1_std |   roc_auc_mean |   roc_auc_std |
|:-----------------------|----------------:|---------------:|-----------------:|----------------:|--------------:|-------------:|----------:|---------:|---------------:|--------------:|
| Majority class         |          0.4693 |         0      |           0.4693 |          0      |        1      |       0      |    0.6389 |   0      |         0.5    |        0      |
| Previous-day direction |          0.4962 |         0      |           0.4631 |          0      |        0.4612 |       0      |    0.4622 |   0      |         0.4942 |        0      |
| Logistic Regression    |          0.4674 |         0      |           0.4391 |          0      |        0.4857 |       0      |    0.4612 |   0      |         0.4659 |        0      |
| MLP                    |          0.5026 |         0.008  |           0.4743 |          0.0039 |        0.5442 |       0.0872 |    0.5044 |   0.0383 |         0.4906 |        0.0159 |
| RNN                    |          0.4911 |         0.0106 |           0.4595 |          0.0072 |        0.4748 |       0.0548 |    0.4658 |   0.0264 |         0.4949 |        0.0097 |
| LSTM                   |          0.5147 |         0.0154 |           0.4879 |          0.0108 |        0.6259 |       0.1121 |    0.545  |   0.0359 |         0.5123 |        0.0145 |
| GRU                    |          0.5255 |         0.0149 |           0.4966 |          0.016  |        0.5048 |       0.0913 |    0.4971 |   0.0373 |         0.5365 |        0.0032 |
| LSTM (tuned)           |          0.5026 |         0.0178 |           0.4743 |          0.0168 |        0.5578 |       0.1143 |    0.5093 |   0.0519 |         0.5112 |        0.0148 |

## Statistical tests (LSTM tuned, seed-ensemble)

- binom_p_vs_coinflip: p = 0.5522
- binom_p_vs_majority: p = 0.9375
- mcnemar_p_lstm_vs_mlp: p = 0.3786
- mcnemar_p_lstm_vs_logreg: p = 0.3349

## Ablations

| experiment            |   window |   accuracy |    auc |
|:----------------------|---------:|-----------:|-------:|
| window=10             |       10 |     0.5019 | 0.4985 |
| window=20             |       20 |     0.4904 | 0.5071 |
| window=30             |       30 |     0.4943 | 0.5003 |
| window=60             |       60 |     0.5038 | 0.5157 |
| features=raw OHLCV+MA |       30 |     0.4693 | 0.481  |
| scaler=minmax         |       30 |     0.4693 | 0.4889 |
| scaler=standard       |       30 |     0.4943 | 0.5003 |
| features=engineered   |       30 |     0.4943 | 0.5003 |

## Walk-forward by year

|   year |   n_test |   lstm_acc |   lstm_auc |   majority_acc |
|-------:|---------:|-----------:|-----------:|---------------:|
|   2021 |      261 |     0.4521 |     0.4258 |         0.5057 |
|   2022 |      260 |     0.4923 |     0.5143 |         0.5231 |
|   2023 |      260 |     0.5385 |     0.5515 |         0.5346 |
|   2024 |      262 |     0.5153 |     0.5387 |         0.5382 |
|   2025 |      260 |     0.4808 |     0.4936 |         0.5231 |

## Hyper-parameter search (validation only)

| units    |   dropout |     lr |   val_loss |   val_auc |
|:---------|----------:|-------:|-----------:|----------:|
| (32, 16) |       0.2 | 0.001  |     0.6906 |    0.5707 |
| (32, 16) |       0.4 | 0.001  |     0.6907 |    0.5677 |
| (32, 16) |       0.2 | 0.0005 |     0.6917 |    0.5643 |
| (32, 16) |       0.4 | 0.0005 |     0.6923 |    0.5545 |
| (64, 32) |       0.4 | 0.001  |     0.693  |    0.5463 |
| (64, 32) |       0.4 | 0.0005 |     0.6935 |    0.5515 |
| (64, 32) |       0.2 | 0.001  |     0.6939 |    0.536  |
| (64, 32) |       0.2 | 0.0005 |     0.694  |    0.548  |

## Permutation importance (top 8)

| feature    |   auc_drop |    std |
|:-----------|-----------:|-------:|
| bb_width   |     0.0102 | 0.0083 |
| ret_5      |     0.004  | 0.0031 |
| ma10_ratio |     0.003  | 0.0067 |
| macd_hist  |     0.0018 | 0.0049 |
| volume_chg |     0.0002 | 0.0014 |
| ma5_ratio  |     0      | 0.0026 |
| ret_1      |    -0.0014 | 0.0023 |
| oc_ret     |    -0.0025 | 0.0045 |

## Backtest (test period)

|          |   total_return |   sharpe |   max_drawdown |
|:---------|---------------:|---------:|---------------:|
| strategy |         0.0988 |   0.5226 |        -0.109  |
| buy_hold |         0.0002 |   0.0653 |        -0.1507 |

Share of days the model was long: 52.7%
