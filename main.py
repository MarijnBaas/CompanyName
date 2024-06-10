# Importing librariespi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from rdkit import Chem
from rdkit.Chem import Draw

file = 'tested_molecules.csv'
molecule_data = pd.read_csv(file)

smiles_list = molecule_data['SMILES'].tolist()

molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list[:5]]

plt.figure(figsize=(100, 100))
for i, mol in enumerate(molecules):
    plt.subplot(1, len(molecules), i + 1)
    img = Draw.MolToImage(mol)
    plt.imshow(img)
    plt.axis('off')

plt.show()

