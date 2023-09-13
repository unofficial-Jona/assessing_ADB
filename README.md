# Assessing aggressive driving behaviour using attentino based models

The findings from this project are currently under review for publication at the BNAIC2023. A link to the paper will be proivded ASAP.

Dataset: https://gamma.umd.edu/researchdirections/autonomousdriving/meteor/

This repository contains the code needed to replicate the experiments conducted for my master thesis. The fulltext version of whcih can be found in [thesis.pdf](https://github.com/unofficial-Jona/thesis/blob/main/thesis.pdf)

The findings presented here build heavily upon the [OadTR](https://github.com/wangxiang1230/OadTR) and [Colar model](https://github.com/VividLe/Online-Action-Detection). Both of which present successfull applications of the attention mechanism in the context of online action detection.
This reserach presents two novel contributions. 
- An intetegrated model architecture is presented that combines architectural concepts from the OadTR model and the Colar model.
- An explainability approach, leveraging prior frame information to generate salient cues. 

For replication please use the YAML file to set up the conda environment.
```
conda env create -f thesis_conda.yml
```

Anyone building uppon this is heavily encouraged to train a custom backbone, as this appeared to be the main performance bottleneck encountered during this project.
