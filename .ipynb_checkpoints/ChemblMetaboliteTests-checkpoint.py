#!/usr/bin/env python

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from MetaboliteFunctions import makeModel, loopedKfoldCrossVal, mixedCV

nonMetaboliteData = 'training_data.csv'
metaboliteData = 'test_data.csv'

descriptors = ["RDKit", "Morgan", "Both"]
trainSet = [[nonMetaboliteData, metaboliteData], [metaboliteData, nonMetaboliteData]]
model = ["RF", "XGBR", "SVR"]

for train, test in trainSet:
    if train == nonMetaboliteData:
       trainName = Non-Natural
       testName = Natural
    elif train == metaboliteData:
       trainName = Natural
       testName = Non-Natural
    for descr in descriptors:
       for mod in model:
            makeModel(train, test, descr, mod, f"Train-{trainName}-Test-{testName}")

for descr in descriptors:
    for mod in model:
        mixedCV("Mixture_Set.csv", descr, model)
