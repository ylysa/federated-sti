import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

new_data_header = ""
with open("DELL.chr22.genotypes.for.modeling.vcf", 'r') as f_in:
    for line_num in range(70):
      f_in.readline()
    new_data_header = f_in.readline()
genotypes = pd.read_csv("DELL.chr22.genotypes.for.modeling.vcf", comment='#', sep='\t', names=new_data_header.strip().split('\t'), header=1, index_col='Sample_id', dtype={'Sample_id':str})
pedigree = pd.read_csv('integrated_call_samples.20130502.ALL.ped', sep='\t', index_col='Individual ID')
X = genotypes.replace({
    '0|0': 0,
    '0|1': 1,
    '1|0': 2,
    '1|1': 3
})

region_mapping = {
    'CHB': 'EAS',
    'CHD': 'EAS',
    'JPT': 'EAS',
    'KHV': 'EAS',
    'CHS': 'EAS',
    'CDX': 'EAS',
    'ACB': 'AFR',
    'LWK': 'AFR',
    'YRI': 'AFR',
    'GWD': 'AFR',
    'MSL': 'AFR',
    'ESN': 'AFR',
    'CEU': 'EUR',
    'GBR': 'EUR',
    'FIN': 'EUR',
    'IBS': 'EUR',
    'TSI': 'EUR',
    'MXL': 'AMR',
    'PUR': 'AMR',
    'CLM': 'AMR',
    'PEL': 'AMR',
    'ASW': 'AMR',
    'GIH': 'SAS',
    'ITU': 'SAS',
    'BEB': 'SAS',
    'STU': 'SAS',
    'PJL': 'SAS',
}

pedigree['Region'] = pedigree['Population'].map(region_mapping)
unique_regions = pedigree['Region'].unique()
combined_test_X = pd.DataFrame()
combined_test_Y = pd.Series()
for region in unique_regions:
    print(region)
    current_region_indices = pedigree[pedigree['Region'] == region].index
    current_X = X[X.index.isin(current_region_indices)]
    current_Y_train = pedigree.loc[current_X.index]['Population']
    train_X, test_X, train_Y, test_Y = train_test_split(current_X, current_Y_train, test_size=0.1, random_state=42)
    train_X.to_csv(f"{region}_X_train.csv")
    train_Y.to_csv(f"{region}_Y_train.csv")
    combined_test_X = pd.concat([combined_test_X, test_X], ignore_index=True)
    combined_test_Y = pd.concat([combined_test_Y, test_Y], ignore_index=True)

combined_test_X.to_csv("test_X.csv")
combined_test_Y.to_csv("test_Y.csv")