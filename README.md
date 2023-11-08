<div align="center">

# YOLO3D: 3D Object Detection with YOLO
</div>

## Introduction

YOLO3D is inspired by [Mousavian et al.](https://arxiv.org/abs/1612.00496) in their paper **3D Bounding Box Estimation Using Deep Learning and Geometry**. YOLO3D uses a different approach, we use 2d gt label result as the input of first stage detector, then use the 2d result as input to regressor model.

## Quickstart
```bash
git clone git@github.com:ApolloAuto/apollo-model-yolo3d.git
```

### creat env for YOLO3D
```shell
cd apollo-model-yolo3d

conda create -n apollo_yolo3d python=3.8 numpy
conda activate apollo_yolo3d
pip install -r requirements.txt
```

### datasets
here we use KITTI data to train. You can download KITTI dataset from [official website](http://www.cvlibs.net/datasets/kitti/). After that, extract dataset to `data/KITTI`.

```shell
ln -s /your/KITTI/path data/KITTI
```

```bash
├── data
│   └── KITTI
│       ├── calib
│       ├── images_2
│       └── labels_2
```
modify [datasplit](data/datasplit.py) file to split train and val data customerly.

```shell
cd data 
python datasplit.py
```

### train
modify [train.yaml](configs/train.yaml) to train your model.

```shell
python src/train.py experiment=sample
```
> log path:    /logs  \
> model path:  /weights

### covert
modify [convert.yaml](configs/convert.yaml) file to trans .ckpt to .pt model

```shell
python convert.py
```

### inference
In order to show the real model infer ability, we crop image according to gt 2d box as yolo3d input, you can use following command to plot 3d results.

modify [inference.yaml](configs/inference.yaml) file to change .pt model path.
**export_onnx=True** can export onnx model.

```shell
python inference.py \
          source_dir=./data/KITTI \
          detector.classes=6 \
          regressor_weights=./weights/pytorch-kitti.pt \
          export_onnx=False \
          func=image
```

- source_dir:             path os datasets, include /image_2 and /label_2 folder                    
- detector.classes:       kitti class
- regressor_weights:      your model
- export_onnx:            export onnx model for apollo

> result path: /outputs

### evaluate
generate label for 3d result:
```shell
python inference.py \
          source_dir=./data/KITTI \
          detector.classes=6 \
          regressor_weights=./weights/pytorch-kitti.pt \
          export_onnx=False \
          func=label
```
> result path: /data/KITTI/result

```bash
├── data
│   └── KITTI
│       ├── calib
│       ├── images_2
│       ├── labels_2
│       └── result
```

modify label_path、result_path and label_split_file in [kitti_object_eval](kitti_object_eval) folder script run.sh, with the help of it we can calculate mAP:
```shell
cd kitti_object_eval
sh run.sh
```

## Acknowledgement
- [yolo3d-lighting](https://github.com/ruhyadi/yolo3d-lightning)
- [skhadem/3D-BoundingBox](https://github.com/skhadem/3D-BoundingBox)
- [Mousavian et al.](https://arxiv.org/abs/1612.00496)
```
@misc{mousavian20173d,
      title={3D Bounding Box Estimation Using Deep Learning and Geometry}, 
      author={Arsalan Mousavian and Dragomir Anguelov and John Flynn and Jana Kosecka},
      year={2017},
      eprint={1612.00496},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```