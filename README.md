# FeatureCloud App 

### Run federated-sti

#### Prerequisite

To run federated-sti, you should install Docker and FeatureCloud pip package:

```shell
pip install featurecloud
```

Then either download federated-sti image from the FeatureCloud docker repository:

```shell
featurecloud app download featurecloud.ai/federated-sti
```

Or build the app locally:

```shell
featurecloud app build featurecloud.ai/federated-sti
```

#### Run federated-sti in the test-bed

You can run federated-sti as a standalone app in the [FeatureCloud test-bed](https://featurecloud.ai/development/test) or [FeatureCloud Workflow](https://featurecloud.ai/projects). You can also run the app using CLI:

```shell
featurecloud test start --app-image featurecloud.ai/federated-sti --client-dirs=client1, client2, client3
```

## Model Architecture

The architecture of the Split-Transformer Impute (STI) model used in this project is based on the work by Mowlaei, Li, Chen, Jamialahmadi, Kumar, Rebbeck and Shi [1].

### References
<a id="1">[1]</a>
Mohammad Erfan Mowlaei, Chong Li, Junjie Chen, Benyamin Jamialahmadi, Sudhir Kumar, Timothy Richard Rebbeck, Xinghua Shi, 2023.
bioRxiv 2023.03.05.531190; doi: https://doi.org/10.1101/2023.03.05.531190
