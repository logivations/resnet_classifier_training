# resnet_classifier_training
Training simple resnet classifier for faurecia.

### Setup
- create python3.8 env
- `pip install -r requirements.txt`

### Training:
`python train.py --train_path /path/to/root_dir` 

### Faurecia UC2: Folder structure

```bash
.
├── filled
│   ├── slot1
│   │   ├── crop1.jpg
│   │   └── crop2.jpg
│   └── slot2
│       ├── crop1.jpg
│       └── crop2.jpg
└── unfilled
    ├── slot1
    │   ├── crop1.jpg
    │   └── crop2.jpg
    └── slot2
        ├── crop1.jpg
        └── crop2.jpg

```


