import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("tested_molecules.csv")

# Function to compute physicochemical properties
def compute_properties(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    properties = {
        'Ipc': Descriptors.Ipc(molecule),
        'BertzCT': Descriptors.BertzCT(molecule),
        'MolLogP': Descriptors.MolLogP(molecule),
        'MolWt': Descriptors.MolWt(molecule),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(molecule),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(molecule),
        'NumHAcceptors': Descriptors.NumHAcceptors(molecule),
        'NumHDonors': Descriptors.NumHDonors(molecule),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
        'TPSA': Descriptors.TPSA(molecule),
        'LabuteASA': Descriptors.LabuteASA(molecule),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(molecule),
        'NHOHCount': Descriptors.NHOHCount(molecule),
        'NOCount': Descriptors.NOCount(molecule),
        'FractionCSP3': Descriptors.FractionCSP3(molecule),
        'RingCount': Descriptors.RingCount(molecule),
        'MolMR': Descriptors.MolMR(molecule),
        'ExactMolWt': Descriptors.ExactMolWt(molecule),
        'MaxPartialCharge': Descriptors.MaxPartialCharge(molecule),
        'MinPartialCharge': Descriptors.MinPartialCharge(molecule),
        'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge(molecule),
        'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge(molecule),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molecule),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(molecule),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(molecule),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule),
        'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        'NumAtomStereoCenters': rdMolDescriptors.CalcNumAtomStereoCenters(molecule),
        'NumUnspecifiedAtomStereoCenters': rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(molecule),
        'Chi0v': Descriptors.Chi0v(molecule),
        'Chi1v': Descriptors.Chi1v(molecule),
        'Chi2v': Descriptors.Chi2v(molecule),
        'Kappa1': Descriptors.Kappa1(molecule),
        'Kappa2': Descriptors.Kappa2(molecule),
        'Kappa3': Descriptors.Kappa3(molecule),
        'MaxEStateIndex': Descriptors.MaxEStateIndex(molecule),
        'MinEStateIndex': Descriptors.MinEStateIndex(molecule),
        'MaxAbsEStateIndex': Descriptors.MaxAbsEStateIndex(molecule),
        'MinAbsEStateIndex': Descriptors.MinAbsEStateIndex(molecule),
    }
    return properties



# Compute properties for each molecule
properties_df = data['SMILES'].apply(lambda x: pd.Series(compute_properties(x)))

# Combine properties with original data
data = pd.concat([data, properties_df], axis=1)

# Separate features and labels
X = data.drop(columns=['SMILES', 'PKM2_inhibition', 'ERK2_inhibition'])
y_PKM2 = data['PKM2_inhibition']
y_ERK2 = data['ERK2_inhibition']

# Calculate mutual information for PKM2
mi_PKM2 = mutual_info_classif(X, y_PKM2)

# Calculate mutual information for ERK2
mi_ERK2 = mutual_info_classif(X, y_ERK2)

# Train Random Forest for PKM2
rf_PKM2 = RandomForestClassifier()
rf_PKM2.fit(X, y_PKM2)

# Train Random Forest for ERK2
rf_ERK2 = RandomForestClassifier()
rf_ERK2.fit(X, y_ERK2)

# Get feature importances from Random Forest
rf_importances_PKM2 = rf_PKM2.feature_importances_
rf_importances_ERK2 = rf_ERK2.feature_importances_

# Create DataFrame to hold feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_PKM2': mi_PKM2,
    'MI_ERK2': mi_ERK2,
    'RF_PKM2': rf_importances_PKM2,
    'RF_ERK2': rf_importances_ERK2
})

# Sort and print results for PKM2
feature_importance_df.sort_values(by='MI_PKM2', ascending=False, inplace=True)
print("Feature Importance for PKM2:")
print(feature_importance_df[['Feature', 'MI_PKM2', 'RF_PKM2']])

# Sort and print results for ERK2
feature_importance_df.sort_values(by='MI_ERK2', ascending=False, inplace=True)
print("Feature Importance for ERK2:")
print(feature_importance_df[['Feature', 'MI_ERK2', 'RF_ERK2']])
