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



### References
<a id="1">[1]</a> 
Matschinske, J., Späth, J., Nasirigerdeh, R., Torkzadehmahani, R., Hartebrodt, A., Orbán, B., Fejér, S., Zolotareva,
O., Bakhtiari, M., Bihari, B. and Bloice, M., 2021.
The FeatureCloud AI Store for Federated Learning in Biomedicine and Beyond. arXiv preprint arXiv:2105.05734.
