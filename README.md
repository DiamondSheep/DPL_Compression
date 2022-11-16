# This is the official repository for IEEE ICDM 2021 paper: **Dictionary Pair-based Data-Free Fast Deep Neural Network Compression**

![Main Process](img_README/process.png)

## How to run
1. add soft link to your dataset folder which should be organized as following: 
   ```
   data
    |
    --imagenet
        |
        --ILSVRC2012_img_val.lmdb
        --ILSVRC2012_img_train.lmdb (unused)
    --cifar10
   ```
   (or modify the source code to fit your environment)
2. run script
    ```
    sh run_DPL_Compress.sh
    ```
    It will make directories to save results and reconstructed path files.
    And pre-trained models from PyTorchcv will be downloaded

Noted that we implement evaluation on ImageNet with [**lmdb**](http://www.lmdb.tech/doc/), the code of [get_imagenet.py](utils/get_imagenet.py) should be modified to fit your environment.

## Further Improvement

This conference paper is invited for [*Knowledge and Information Systems (KAIS)*](https://www.springer.com/journal/10115) publication as ***Best-ranked*** paper. New features in KAIS verison are listed as following:

- Shared dictionary design
- Auto hyper-parameter tuning 
- Higher compression ratio

