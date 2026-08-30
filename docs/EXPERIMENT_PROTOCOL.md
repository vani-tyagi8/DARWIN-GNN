# Controlled experiment protocol

## Fixed factors

- Devign architecture, optimizer, learning rate, batch size, epoch budget, patience,
  preprocessing, graph vocabulary, split, evaluation threshold, and seed list.
- Untouched validation and test distributions.
- Validation PR-AUC for checkpoint selection; no test-set tuning.

## Independent variable

Exactly one of: baseline, random oversampling, random undersampling,
class-weighted BCE, or focal loss.

Oversampling duplicates minority graph indices with replacement. Undersampling draws
majority graph indices without replacement. Class weighting uses `N_negative/N_positive`
as `pos_weight`. Focal loss defaults to gamma 2; gamma and alpha must be fixed before
test evaluation and reported.

## Repetition and reporting

Use seeds 42, 1337, and 2026 at minimum. Report every run, plus mean and standard
deviation for Precision, Recall, F1, MCC, and PR-AUC. Retain confusion-matrix counts
and per-function probabilities. Compare strategies within each dataset before making
cross-dataset claims.

Accuracy may be recorded for diagnostics but is not a primary outcome. Do not select a
strategy from a single seed or a single metric. Treat a result as an improvement only
when its minority-class benefit is visible across repetitions and its precision-recall
cost is reported.

