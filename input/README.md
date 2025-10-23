## Overview

This directory contains the input data used for training and inference.

## Training Input

Training data are organized into folders, where each folder name represents a class label.

```
input/
└── train/
    ├── NC/       # Necrosis
    │   ├── img_001.jpg
    │   ├── img_002.jpg
    │   └── ...
    ├── CT/       # Cellular Tumor
    │   ├── img_003.jpg
    │   ├── img_004.jpg
    │   └── ...
    └── ...       # Other classes
```

## Validation Input

Validation data are organized similarly to training data.

```
input/
└── valid/
    ├── NC/       # Necrosis
    │   ├── img_001.jpg
    │   ├── img_002.jpg
    │   └── ...
    ├── CT/       # Cellular Tumor
    │   ├── img_003.jpg
    │   ├── img_004.jpg
    │   └── ...
    └── ...       # Other classes
```

## Test Input

Test data consist of individual image files without labels.

```
input/
└── test/
    ├── sample_01.jpg
    ├── sample_02.png
    └── sample_03.jpg
```
