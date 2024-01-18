# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

new_data_header = ""
with open("client.vcf", 'r') as f_in:
    for line_num in range(70):
      f_in.readline()
    new_data_header = f_in.readline()
genotypes = pd.read_csv("client.vcf", comment='#', sep='\t', names=new_data_header.strip().split('\t'), header=1, index_col='Sample_id', dtype={'Sample_id':str})
pedigree = pd.read_csv('integrated_call_samples.20130502.ALL.ped', sep='\t', index_col='Individual ID')
Y_train = pedigree.loc[genotypes.index]['Population']
X = genotypes[genotypes.index.isin(Y_train.index)]
X = X.replace({
    '0|0': 0,
    '0|1': 1,
    '1|0': 2,
    '1|1': 3
})

X.to_csv('X.csv')
Y_train.to_csv('Y_train.csv')