# Description

Federated learning for imputing SNP data. The architecture of the Split-Transformer Impute (STI) model used in this project is based on the work by Mowlaei, Li, Chen, Jamialahmadi, Kumar, Rebbeck and Shi [1].

# Data
The DELL.chr22.genotypes.for.modeling.vcf dataset can be downloaded from https://github.com/Mycheaux/STI/blob/main/data/STI_benchmark_datasets.zip.

# Input
The model does not rely on a specific file format, as long as the inputs are one-hot encoded. Otherwise, preprocessing is required, similar to the preprocessing.py file.

# Output
The imputation results are saved in CSV files.
# Run fed-impute-sequencing-learner

#### Prerequisite

To run fed-impute-sequencing-learner, you should install Docker and FeatureCloud pip package:

```shell
pip install featurecloud
```

Then either download fed-impute-sequencing-learner image from the FeatureCloud docker repository:

```shell
featurecloud app download featurecloud.ai/fed-impute-sequencing-learner
```

Or build the app locally:

```shell
featurecloud app build featurecloud.ai/fed-impute-sequencing-learner
```

#### Run fed-impute-sequencing-learner in the test-bed

You can run fed-impute-sequencing-learner as a standalone app in the [FeatureCloud test-bed](https://featurecloud.ai/development/test) or [FeatureCloud Workflow](https://featurecloud.ai/projects). You can also run the app using CLI:

```shell
featurecloud test start --app-image featurecloud.ai/fed-impute-sequencing-learner --client-dirs=client1, client2, client3
```

### References
<a id="1">[1]</a>
Mohammad Erfan Mowlaei, Chong Li, Junjie Chen, Benyamin Jamialahmadi, Sudhir Kumar, Timothy Richard Rebbeck, Xinghua Shi, 2023.
bioRxiv 2023.03.05.531190; doi: https://doi.org/10.1101/2023.03.05.531190