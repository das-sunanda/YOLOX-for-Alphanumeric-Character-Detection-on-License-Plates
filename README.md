# YOLOX for Alphanumeric Character Detection on License Plates

## Overview
This repository contains the implementation and training pipeline for an alphanumeric character detection model based on YOLOX. The model is designed to localize and classify alphanumeric characters from license plates, leveraging a dataset derived from the UC3M-LP1 dataset, following the COCO annotation format.

## Features
- Uses YOLOX-S for balanced accuracy and computational efficiency
- Trained on a diverse character detection dataset
- Augmentations applied: mosaic, mixup, rotation, translation, scaling, shear
- Optimized with Adam optimizer
- Trained for 200 epochs on an Nvidia A100 GPU

## Dataset
The character detection dataset is derived from the UC3M-LP1 dataset, which provides annotated license plate images for European (Spanish) vehicles.
- **Total unique vehicles:** 2,547
- **Total annotated characters:** 12,757
- **Classes:** 0-9, A-Z (36 classes)
- **Variations:** Different lighting conditions, angles, occlusions
- **Format:** COCO dataset structure

Dataset source: [UC3M-LP Dataset](https://lsi.uc3m.es/2023/12/22/uc3m-lp-a-new-open-source-dataset/)

## Installation
```bash
# Clone the repository
git clone https://github.com/das-sunanda/YOLOX-for-Alphanumeric-Character-Detection-on-License-Plates.git
cd YOLOX-for-Alphanumeric-Character-Detection-on-License-Plates

# Install dependencies
pip install -r requirements.txt

# Setup YOLOX
cd YOLOX
python setup.py build develop
```

## Training
```bash
python tools/train.py -f exps/default/yolox_s.py -d 1 -b 16 --fp16 -o
```

## Evaluation
```bash
python tools/eval.py -f exps/default/yolox_s.py -c yolox_s.pth -b 16 -d 1 --conf 0.01 --nms 0.65
```

## Performance Metrics
| Metric | Score |
|--------|-------|
| AP@[IoU=0.50:0.95] | 0.777 |
| AP@[IoU=0.50] | 0.990 |
| AP@[IoU=0.75] | 0.958 |
| AR@[IoU=0.50:0.95] | 0.817 |

### Per-Class Accuracy
| Class | AP | Class | AP | Class | AP |
|-------|----|-------|----|-------|----|
| 0 | 80.554 | 1 | 68.107 | 2 | 77.374 |
| 3 | 79.659 | 4 | 77.304 | 5 | 81.441 |
| ... | ... | ... | ... | ... | ... |

## Citation
If you use this model, please cite the following:

### YOLOX Paper
```bibtex
@article{ge2021yolox,
  title={Yolox: Exceeding yolo series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```

### UC3M-LP Dataset
The dataset is derived from the UC3M-LP dataset:
[UC3M-LP Dataset](https://lsi.uc3m.es/2023/12/22/uc3m-lp-a-new-open-source-dataset/)

```bibtex
@article{uc3m-lp,
  title={UC3M-LP: A New Open-Source Dataset for License Plate Detection and Recognition},
  author={UC3M-LP Team},
  journal={LSI UC3M},
  year={2023},
  url={https://lsi.uc3m.es/2023/12/22/uc3m-lp-a-new-open-source-dataset/}
}
```

## License
This project is licensed under the MIT License.
