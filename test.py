from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('tested_molecules.csv')
smiles_list = df['SMILES'].values
erk2_inhibition = df['ERK2_inhibition'].values  # Binary values for ERK2 inhibition
pkm2_inhibition = df['PKM2_inhibition'].values  # Binary values for PKM2 inhibition

# Calculate descriptors
descriptor_names = [d[0] for d in Descriptors._descList]
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
descriptor_matrix = []

for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol:  # Check if the molecule could be parsed successfully
        descriptor_matrix.append(descriptor_calculator.CalcDescriptors(mol))

# Convert to DataFrame
descriptors_df = pd.DataFrame(descriptor_matrix, columns=descriptor_names)
descriptors_df = descriptors_df.clip(lower=0)

# Feature selection using SelectKBest
X_new_erk2 = SelectKBest(chi2, k='all').fit(descriptors_df, erk2_inhibition)
X_new_pkm2 = SelectKBest(chi2, k='all').fit(descriptors_df, pkm2_inhibition)

# Get the scores and p-values
scores_erk2 = X_new_erk2.scores_
pvalues_erk2 = X_new_erk2.pvalues_
scores_pkm2 = X_new_pkm2.scores_
pvalues_pkm2 = X_new_pkm2.pvalues_

# Combine scores and p-values with descriptor names
erk2_feature_importance = pd.DataFrame({'Descriptor': descriptor_names, 'Score': scores_erk2, 'P-value': pvalues_erk2})
pkm2_feature_importance = pd.DataFrame({'Descriptor': descriptor_names, 'Score': scores_pkm2, 'P-value': pvalues_pkm2})

# Sort the descriptors by score
erk2_sorted_features = erk2_feature_importance.sort_values(by='Score', ascending=False)
pkm2_sorted_features = pkm2_feature_importance.sort_values(by='Score', ascending=False)

# Print the top descriptors
print("Top descriptors for ERK2 inhibition:")
print(erk2_sorted_features.head(20))
print("\nTop descriptors for PKM2 inhibition:")
print(pkm2_sorted_features.head(20))
