# Dataset setup notes

## D-Fire

D-Fire already uses normalized YOLO annotations. Class IDs are expected to map to:

```text
0 fire
1 smoke
```

The pipeline recursively searches under `images_dir` and mirrors each image's relative path under `labels_dir`.

Example:

```text
images_dir: data/raw/dfire/images
labels_dir: data/raw/dfire/labels

Image: data/raw/dfire/images/train/folder/a.jpg
Label: data/raw/dfire/labels/train/folder/a.txt
```

Negative samples may have an empty label file or no label file, depending on the downloaded release. The D-Fire source is therefore configured with `missing_label_is_negative: true`.

## DFS

DFS is expected in Pascal VOC format. Each image must have an XML file with the same relative path and filename stem.

```text
Image: data/raw/dfs/images/a.jpg
XML:   data/raw/dfs/annotations/a.xml
```

The converter reads each `<object>`, maps `fire` and `smoke`, repairs small boundary errors when allowed, and writes normalized YOLO labels.

## Multiple existing splits

Data Prep creates a fresh leakage-controlled split. If your download has separate train/val/test folders under the same `images` and `labels` roots, all of them will be discovered and re-split.

For a strict independent publisher test split, configure it as a separate future evaluation dataset rather than including it in Data Prep training data.

## Adding team-recorded negative images

Add a source with no labels and explicitly enable negative handling:

```yaml
- name: team_negatives
  format: yolo
  images_dir: data/raw/team_negatives/images
  labels_dir: data/raw/team_negatives/labels
  class_map:
    0: 0
    1: 1
  missing_label_is_negative: true
```

Use only genuinely safe scenes. Do not create uncontrolled real fires for data collection. Record non-fire confusers such as lights, sunsets, steam, clouds, red/orange objects, reflections, and screens displaying fire footage.
