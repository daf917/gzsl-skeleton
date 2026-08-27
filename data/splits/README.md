# Dataset Split Files

Each split is released as both JSON and CSV. JSON files are used by code; CSV files are for quick inspection.

- `random_split_1..3`: deterministic random splits generated with the seeds recorded in the JSON files.
- `provided_split_1..3`: explicit provided-protocol class partitions for datasets evaluated under the provided protocol.

Labels are zero-based in `class_id`, matching the training code. `one_based_label` is included for papers and dataset documentation that label classes from 1.
