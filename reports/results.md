# Results: TCS.NS next-day direction (2015-01-01 to 2025-12-31)

Class balance (share of UP days): train 52.2%, val 50.3%, test 45.6%. Best constant guess on test (majority class, known only in hindsight) = 54.4%; this is the strictest baseline. The 'Majority class' row is the achievable version that uses the train-period majority.

## Model comparison (test set, mean over seeds)

| model                  |   accuracy_mean |   accuracy_std |   precision_mean |   precision_std |   recall_mean |   recall_std |   f1_mean |   f1_std |   roc_auc_mean |   roc_auc_std |
|:-----------------------|----------------:|---------------:|-----------------:|----------------:|--------------:|-------------:|----------:|---------:|---------------:|--------------:|
| Majority class         |          0.4564 |         0      |           0.4564 |          0      |        1      |       0      |    0.6267 |   0      |         0.5    |        0      |
| Previous-day direction |          0.5152 |         0      |           0.469  |          0      |        0.4711 |       0      |    0.4701 |   0      |         0.5117 |        0      |
| Logistic Regression    |          0.4929 |         0      |           0.4555 |          0      |        0.5689 |       0      |    0.5059 |   0      |         0.5133 |        0      |
| MLP                    |          0.4767 |         0.0281 |           0.4514 |          0.0173 |        0.6726 |       0.0616 |    0.5393 |   0.0236 |         0.4855 |        0.0399 |
| RNN                    |          0.4692 |         0.0152 |           0.4413 |          0.0146 |        0.6193 |       0.0566 |    0.5151 |   0.029  |         0.4844 |        0.0277 |
| LSTM                   |          0.4604 |         0.0105 |           0.4463 |          0.005  |        0.7585 |       0.0958 |    0.5607 |   0.0271 |         0.4843 |        0.0075 |
| GRU                    |          0.4679 |         0.0245 |           0.4514 |          0.0166 |        0.7733 |       0.0513 |    0.5699 |   0.0254 |         0.4815 |        0.0176 |
| LSTM (tuned)           |          0.4733 |         0.0077 |           0.4487 |          0.0042 |        0.6756 |       0.0693 |    0.5384 |   0.0243 |         0.4894 |        0.0072 |

## Statistical tests (LSTM tuned, seed-ensemble)

- binom_p_vs_coinflip: p = 0.9372
- binom_p_vs_majority: p = 0.9997
- mcnemar_p_lstm_vs_mlp: p = 0.8558
- mcnemar_p_lstm_vs_logreg: p = 0.3950

## Ablations

| experiment            |   window |   accuracy |    auc |
|:----------------------|---------:|-----------:|-------:|
| window=10             |       10 |     0.4665 | 0.4923 |
| window=20             |       20 |     0.4686 | 0.4713 |
| window=30             |       30 |     0.4767 | 0.4928 |
| window=60             |       60 |     0.4767 | 0.4981 |
| features=raw OHLCV+MA |       30 |     0.4564 | 0.5062 |
| scaler=minmax         |       30 |     0.4564 | 0.5176 |
| scaler=standard       |       30 |     0.4767 | 0.4928 |
| features=engineered   |       30 |     0.4767 | 0.4928 |

## Walk-forward by year

|   year |   n_test |   lstm_acc |   lstm_auc |   majority_acc |
|-------:|---------:|-----------:|-----------:|---------------:|
|   2021 |      248 |     0.5403 |     0.5548 |         0.5484 |
|   2022 |      248 |     0.5323 |     0.5455 |         0.5121 |
|   2023 |      245 |     0.551  |     0.5571 |         0.5184 |
|   2024 |      246 |     0.5    |     0.4972 |         0.5163 |
|   2025 |      247 |     0.4696 |     0.5148 |         0.5709 |

## Hyper-parameter search (validation only)

| units    |   dropout |     lr |   val_loss |   val_auc |
|:---------|----------:|-------:|-----------:|----------:|
| (64, 32) |       0.4 | 0.001  |     0.6902 |    0.5547 |
| (32, 16) |       0.4 | 0.001  |     0.6908 |    0.5475 |
| (32, 16) |       0.4 | 0.0005 |     0.6911 |    0.5455 |
| (32, 16) |       0.2 | 0.001  |     0.6916 |    0.5451 |
| (32, 16) |       0.2 | 0.0005 |     0.6916 |    0.5487 |
| (64, 32) |       0.4 | 0.0005 |     0.692  |    0.5481 |
| (64, 32) |       0.2 | 0.0005 |     0.6924 |    0.5378 |
| (64, 32) |       0.2 | 0.001  |     0.6926 |    0.5296 |

## Permutation importance (top 8)

| feature    |   auc_drop |    std |
|:-----------|-----------:|-------:|
| ma20_ratio |     0.0066 | 0.0067 |
| hl_range   |     0.0057 | 0.0028 |
| ma10_ratio |     0.0055 | 0.0043 |
| ma5_ratio  |     0.0033 | 0.0023 |
| ret_5      |     0.0015 | 0.0044 |
| ret_1      |     0.0013 | 0.0023 |
| volume_chg |     0      | 0.0008 |
| vol_10     |    -0.0005 | 0.0019 |

## Backtest (test period)

|          |   total_return |   sharpe |   max_drawdown |
|:---------|---------------:|---------:|---------------:|
| strategy |        -0.0173 |   0.034  |        -0.2649 |
| buy_hold |        -0.1016 |  -0.1652 |        -0.3445 |

Share of days the model was long: 68.6%
