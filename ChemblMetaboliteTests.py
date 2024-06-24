#!/usr/bin/env python

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from MetaboliteFunctions import makeModel, loopedKfoldCrossVal, mixedCV

targetName = 'Acetyl'

nonMetaboliteData = 'training_data.csv'
metaboliteData = 'test_data.csv'

descriptors = ["RDKit", "Morgan", "Both"]
trainSet = [[nonMetaboliteData, metaboliteData], [metaboliteData, nonMetaboliteData]]
model = ["RF", "XGBR", "SVR"]

# Examining Train/Test Sets
for train, test in trainSet:
    if train == nonMetaboliteData:
       trainName = Non-Natural
       testName = Natural
    elif train == metaboliteData:
       trainName = Natural
       testName = Non-Natural
    for descr in descriptors:
       for mod in model:
            df_ret =  makeModel(train, test, descr, mod, f"{targetName}-Train-{trainName}-Test-{testName}")
            All_Models = pd.concat([All_Models, df_ret])
All_Models.to_csv(f"{targetName}-Models.csv", index = True)

# Examining Mixture Sets
for descr in descriptors:
    for mod in model:
        mixedCV("Mixture_Set.csv", descr, model)
