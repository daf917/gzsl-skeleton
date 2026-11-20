# Reproducibility Materials

Use the class manifests in [data/splits](../data/splits) and the prepared files described below. The manuscript results in the README are the reported values. Repeating those runs requires the original sample assignments and skeleton inputs.

## Skeleton preparation

For raw inputs, provide a JSON object mapping sample IDs to `train`, `val`, or `test`. This sample partition is separate from the seen/unseen class manifest. Each GZSL test set must include held-out samples for both groups.

| Dataset | Input | Sample ID |
|---|---|---|
| NTU60 and NTU120 | NTU .skeleton files; the image-plane color coordinates are used | Relative filename stem, e.g. `S001C001P001R001A001` |
| PKU-MMD | 3D recordings, official temporal annotation files, and RGB projection matrices | Relative recording stem and one-based annotation row, e.g. `0001-L:1` |
| UCF101, HMDB-51 and SSv2 | Frame-by-frame HRNet 2D pose JSON with 17 COCO-order joints | Relative class and clip folders, e.g. `0_action/clip01` |

Prepare one dataset/split NPZ, for example:

```bash
python scripts/preprocess_skeletons.py --format ntu --input path/to/nturgbd_skeletons --sample-splits ntu60_samples.json --split-file data/splits/ntu60/random_split_1.json --output data/ntu60/random_split_1.npz --no-attributes
```

For PKU-MMD use `--format pku_mmd` and pass `--annotations`, `--camera-matrix`, and the frame numbering with `--frame-index-base`. For pose exports use `--format hrnet_json`. Prefix each class folder with its zero-based class ID and put one JSON file per frame in each clip directory. The preprocessing script normalizes joint coordinates by body center and scale, then uniformly samples 64 frames.

For RGB inputs, the optional MMPose 1.x HRNet entry point accepts a COCO HRNet model configuration and weights. Supply person boxes for full-scene frames.

```bash
python scripts/extract_hrnet_poses.py --input path/to/rgb_frames --output path/to/poses --config path/to/coco_hrnet.py --weights path/to/hrnet.pth --boxes person_boxes.json
```

Use class/clip folders and one image per frame. The box JSON maps each relative image path to its list of XYXY person boxes. Pass the resulting pose folders to `--format hrnet_json`.

To use existing 2D sequences, `--format npz` accepts arrays `skeletons` of shape `[N, T, J, 2]`, `labels` with zero-based IDs, and either a `split` array (0 train, 1 validation, 2 test) or a membership file using `sample_ids`. Recognition training fits motion-attribute statistics on seen training samples and uses those same statistics in evaluation. Omit `--no-attributes` to also export normalized motion attributes for analysis.

## Class descriptions and splits

[The prompt template](../prompts/prompt_template.md) documents offline GPT-4o-mini generation with temperature 0, top-p 1.0, an 80-token limit, and deterministic text normalization. The description JSON files in [data/prompts](../data/prompts) identify each class by `class_id` and provide six body-part descriptions. [Appendix A](appendix_a_supplementary.md) lists diagnostic results and settings.

Use the same class manifest in pretraining, recognition training, and evaluation. See [the README](../README.md#train-and-evaluate-a-split) for the commands and three-split HM aggregation.

To generate descriptions for a new class list, set `OPENAI_API_KEY` and run `python scripts/generate_prompts.py --dataset ssv2 --classes action_classes.json --generate ssv2.json`. The class list can be an ordered JSON array of names or records with explicit zero-based IDs. Use a separate output file to keep the supplied descriptions available.
