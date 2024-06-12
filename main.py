# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

def read_data(file):
    """
    Reads the data from the file
    :param file: file to read
    :return: pandas dataframe
    """
    return pd.read_csv(file)


def draw_molecule(molecule_data):
    """
    Draws the molecules
    :param molecule_data: pandas dataframe
    :return: None
    """
    smiles_list = molecule_data['SMILES'].tolist()

    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list[:5]]

    plt.figure(figsize=(100, 100))
    for i, mol in enumerate(molecules):
        plt.subplot(1, len(molecules), i + 1)
        img = Draw.MolToImage(mol)
        plt.imshow(img)
        plt.axis('off')

    plt.show()

def calculate_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)





# Example
file = 'tested_molecules.csv'
df = read_data('tested_molecules.csv')

# Apply the function to the 'SMILES' column
df['Fingerprints'] = df['SMILES'].apply(calculate_fingerprints)

df['Fingerprints'] = df['Fingerprints'].apply(lambda fp: list(fp))

# Save the dataframe to a new file
df.to_csv('modified_molecules.csv', index=False)

