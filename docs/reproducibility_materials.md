# Reproducibility Materials

This repository includes the reproducibility materials for the paper:

- `prompts/prompt_template.md`: exact Appendix A prompt template.
- `data/prompts/{ntu60,ntu120,ucf101,pku_mmd,hmdb51}.json`: body-part descriptions for each released action class.
- `data/prompts/appendix_a_examples.json`: representative generated examples synchronized with Appendix A Table 6.
- `data/splits/<dataset>/*.json` and `.csv`: explicit seen/unseen class partitions for each split.
- `scripts/preprocess_skeletons.py`: preprocessing utilities for raw skeleton parsing, sequence normalization, resampling, 2D pose conversion, motion-attribute extraction, and train-statistics normalization.
- `docs/appendix_a_supplementary.md`: Appendix A diagnostic metrics, results tables, body-part partition settings, and hyperparameter ablation.

The body-part descriptions follow the paper's offline semantic-generation setting: GPT-4o-mini with temperature `0`, top-p `1.0`, maximum output length `80` tokens, and deterministic text normalization only. No manual semantic rewriting is applied.

The released body-part description JSON files are minimal arrays. Each entry contains only:

```json
{
  "class_id": 0,
  "action_class": "drink water",
  "head": "...",
  "torso": "...",
  "left_arm": "...",
  "right_arm": "...",
  "left_leg": "...",
  "right_leg": "..."
}
```

All split files use zero-based `class_id` values matching Python labels and also include `one_based_label` values for dataset documentation.
