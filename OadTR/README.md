# Code is mainly coppied from the original repository: https://github.com/wangxiang1230/OadTR

# Video Features
OadTR uses RGB and Flow features. to extract these features from the original videos the following models were used:
* RGB: `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`
* FLOW: `torchvision.models.optical_flow.raft_large(weights=Raft_Large_Weights.DEFAULT)`


# OadTR
Code for ICCV2021 paper: "OadTR: Online Action Detection with Transformers" [["Paper"]](https://arxiv.org/pdf/2106.11149.pdf)

## Dependencies
please install environment thesis_conda.yml from parent directory.


# Prepare
* update `custom_dataset.py` so that paths point to correct file locations.
* check `custom_config.py` to make sure default values are correct. Otherwise add the corresponding argument, value pair to the pather when initializing the training process.

# Training
```
python custom_main.py 
```


# Citing OadTR
Please cite our paper in your publications if it helps your research:

```BibTeX
@article{wang2021oadtr,
  title={OadTR: Online Action Detection with Transformers},
  author={Wang, Xiang and Zhang, Shiwei and Qing, Zhiwu and Shao, Yuanjie and Zuo, Zhengrong and Gao, Changxin and Sang, Nong},
  journal={arXiv preprint arXiv:2106.11149},
  year={2021}
}
```
