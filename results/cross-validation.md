# Qwen3 permission decisions on held-out queries

The 1,444 decisions cover 169 queries (871 Yes and 573 No). Each seed (42, 43, 44) assigns whole queries to five folds shared by both models. Each adapter trains on four folds for two epochs with an effective batch size of eight and is evaluated on the fifth fold.

| Model | Weights | Accuracy (%) | Macro-F1 | FP | FN | FPR (%) | FNR (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | Raw | 86.84 | 0.8616 | 106 | 84 | 18.50 | 9.64 |
| Qwen3-4B | SFT | 95.66 ± 0.11 | 0.9546 ± 0.0011 | 34.00 ± 2.65 | 28.67 ± 3.06 | 5.93 ± 0.46 | 3.29 ± 0.35 |
| Qwen3-8B | Raw | 84.97 | 0.8479 | 38 | 179 | 6.63 | 20.55 |
| Qwen3-8B | SFT | 94.76 ± 0.26 | 0.9451 ± 0.0027 | 41.67 ± 1.15 | 34.00 ± 4.58 | 7.27 ± 0.20 | 3.90 ± 0.53 |

Raw denotes the original weights evaluated once on the same 1,444 decisions. For SFT, each repeat pools one out-of-fold prediction per decision. Entries show the mean ± sample standard deviation across three repeats. FP counts incorrect access grants and FN counts incorrect denials. All evaluated outputs were valid Yes/No decisions.

## Training and validation loss

![Qwen3-4B training and validation loss versus optimizer step](cv-4b-loss.png)

![Qwen3-8B training and validation loss versus optimizer step](cv-8b-loss.png)

The loss measurements come from a second run of the same 30 fits, using the original query assignments, seeds, effective batch size of eight and two-epoch schedule. At each marked step, both losses were measured at the same weights with dropout disabled, across all examples in the training folds and the held-out validation fold. Each loss is the mean negative log-likelihood over the answer and end-of-turn tokens. The validation fold was used for monitoring only. All runs used their scheduled final update.

Each point is the arithmetic mean of 15 fold measurements for one model. Shading shows the 25th to 75th percentiles across those fits. Both panels use the same axes. Measurements start before training and repeat every 32 optimizer steps. The plots end at the last shared step (288). Depending on fold size, training ended at step 288, 290 or 292.
