import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from MetaboliteFunctions import makeModel, loopedKfoldCrossVal, mixedCV

nonMetaboliteData = 'Met_Files/Monoamine_oxidase_A_CHEMBL1951_NP.csv'
metaboliteData = 'test_data.csv'

descriptors = ["RDKit", "Morgan", "Both"]
trainSet = [[nonMetaboliteData, metaboliteData], [metaboliteData, nonMetaboliteData]]
model = ["RF", "XGBR", "SVR", "SVRLinear"]

#for train, test in trainSet:
    #for descr in descriptors:
       #for mod in model:
            #makeModel(train, test, descr, mod, f"{train} + {test}")

#for descr in descriptors:
    #for mod in model:
        #mixedCV("Mixture_Set.csv", descr, model)

# Extra Test Stuff:

#mixDf = pd.read_csv("Mixture_Set.csv")

#smiles_strings = mixDf['SMILES'].tolist()
#mySmiles = [Chem.MolFromSmiles(mol) for mol in smiles_strings]
#myDescriptors = [Descriptors.CalcMolDescriptors(mol) for mol in mySmiles]
#df2Mix = pd.DataFrame(myDescriptors, index = mixDf.index)

#allMetabolites = mixDf["Metabolite"].tolist()
#df2Mix["Metabolite"] = allMetabolites
#train_X = df2Mix.dropna(axis = 1)
#train_y = mixDf.pIC50
#metabolites = mixDf.Metabolite
#train_X = train_X.drop("Metabolite", axis = 1)

#loopedKfoldCrossVal("RF", 10, train_X, train_y, "Mixture", metabolites)

makeModel(nonMetaboliteData, nonMetaboliteData, "Both", "RF", "bruh")