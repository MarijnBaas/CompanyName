import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load the dataset
data = pd.read_csv("tested_molecules.csv")

# Function to compute physicochemical properties
def compute_properties(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    properties = {
        'MolWt': Descriptors.MolWt(molecule),
        'Ipc': Descriptors.Ipc(molecule),
        'BertzCT': Descriptors.BertzCT(molecule),
        'MolLogP': Descriptors.MolLogP(molecule),
        'NumHDonors': Descriptors.NumHDonors(molecule),
        'NumHAcceptors': Descriptors.NumHAcceptors(molecule),
        'TPSA': Descriptors.TPSA(molecule),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(molecule),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(molecule),
        'NumRings': rdMolDescriptors.CalcNumRings(molecule),
        'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(molecule),
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(molecule),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molecule),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(molecule),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(molecule),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        'NumHBA': rdMolDescriptors.CalcNumHBA(molecule),
        'NumHBD': rdMolDescriptors.CalcNumHBD(molecule),
        'NumLipinskiHBA': rdMolDescriptors.CalcNumLipinskiHBA(molecule),
        'NumLipinskiHBD': rdMolDescriptors.CalcNumLipinskiHBD(molecule),
        'NumRings': rdMolDescriptors.CalcNumRings(molecule),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(molecule),
        'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molecule),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(molecule),
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(molecule),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molecule),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(molecule),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(molecule),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        'NumHBA': rdMolDescriptors.CalcNumHBA(molecule),
        'NumHBD': rdMolDescriptors.CalcNumHBD(molecule),
        'NumLipinskiHBA': rdMolDescriptors.CalcNumLipinskiHBA(molecule),
        'NumLipinskiHBD': rdMolDescriptors.CalcNumLipinskiHBD(molecule),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(molecule),
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(molecule),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molecule),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(molecule),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(molecule),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        'NumHBA': rdMolDescriptors.CalcNumHBA(molecule),
        'NumHBD': rdMolDescriptors.CalcNumHBD(molecule),
        'NumLipinskiHBA': rdMolDescriptors.CalcNumLipinskiHBA(molecule),
        'NumLipinskiHBD': rdMolDescriptors.CalcNumLipinskiHBD(molecule),
        'NumRings': rdMolDescriptors.CalcNumRings(molecule),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(molecule),
        'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molecule),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(molecule),
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(molecule),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molecule),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(molecule),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(molecule),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        'NumHBA': rdMolDescriptors.CalcNumHBA(molecule),
        'NumHBD': rdMolDescriptors.CalcNumHBD(molecule),
        'NumLipinskiHBA': rdMolDescriptors.CalcNumLipinskiHBA(molecule),
        'NumLipinskiHBD': rdMolDescriptors.CalcNumLipinskiHBD(molecule),
        'NumRings': rdMolDescriptors.CalcNumRings(molecule),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(molecule),
        'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molecule),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(molecule),
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(molecule),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molecule),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(molecule),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(molecule),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        'NumHBA': rdMolDescriptors.CalcNumHBA(molecule),
        'NumHBD': rdMolDescriptors.CalcNumHBD(molecule),
        'NumLipinskiHBA': rdMolDescriptors.CalcNumLipinskiHBA(molecule),
        'NumLipinskiHBD': rdMolDescriptors.CalcNumLipinskiHBD(molecule),
        'NumRings': rdMolDescriptors.CalcNumRings(molecule),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(molecule),
        'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molecule),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(molecule),
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(molecule),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molecule),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(molecule),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(molecule),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        'NumHBA': rdMolDescriptors.CalcNumHBA(molecule),
        'NumHBD': rdMolDescriptors.CalcNumHBD(molecule),
        'NumLipinskiHBA': rdMolDescriptors.CalcNumLipinskiHBA(molecule),
        'NumLipinskiHBD': rdMolDescriptors.CalcNumLipinskiHBD(molecule),
        'NumRings': rdMolDescriptors.CalcNumRings(molecule),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(molecule),
        'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molecule),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(molecule),
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(molecule),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molecule),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(molecule),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(molecule),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        'NumHBA': rdMolDescriptors.CalcNumHBA(molecule),
        'NumHBD': rdMolDescriptors.CalcNumHBD(molecule),
        'NumLipinskiHBA': rdMolDescriptors.CalcNumLipinskiHBA(molecule),
        'NumLipinskiHBD': rdMolDescriptors.CalcNumLipinskiHBD(molecule),
        'NumRings': rdMolDescriptors.CalcNumRings(molecule),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(molecule),
        'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molecule),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(molecule),
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(molecule),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molecule),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(molecule),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(molecule),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        'NumHBA': rdMolDescriptors.CalcNumHBA(molecule),
        'NumHBD': rdMolDescriptors.CalcNumHBD(molecule),
        'NumLipinskiHBA': rdMolDescriptors.CalcNumLipinskiHBA(molecule),
        'NumLipinskiHBD': rdMolDescriptors.CalcNumLipinskiHBD(molecule),
        'NumRings': rdMolDescriptors.CalcNumRings(molecule),
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
        'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(molecule),
        'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molecule),
        'NumRotatableBonds': rdMolDescriptors.CalcNumRotatableBonds(molecule),
        'NumHeterocycles': rdMolDescriptors.CalcNumHeterocycles(molecule),
        'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(molecule),
        'NumAromaticCarbocycles': rdMolDescriptors.CalcNumAromaticCarbocycles(molecule),
        'NumAromaticHeterocycles': rdMolDescriptors.CalcNumAromaticHeterocycles(molecule),
        'NumSaturatedCarbocycles': rdMolDescriptors.CalcNumSaturatedCarbocycles(molecule),
        'NumSaturatedHeterocycles': rdMolDescriptors.CalcNumSaturatedHeterocycles(molecule),
        'NumAliphaticCarbocycles': rdMolDescriptors.CalcNumAliphaticCarbocycles(molecule),
        'NumAliphaticHeterocycles': rdMolDescriptors.CalcNumAliphaticHeterocycles(molecule),
        'NumSpiroAtoms': rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
    }
    return properties

properties_df = data['SMILES'].apply(lambda x: pd.Series(compute_properties(x)))

def prepare_data(data, target):
    # Prepare feature and target variables
    X = data
    y = target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def train_rf_model(X_train, y_train):
    # Train a Random Forest Classifier
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    return rf

X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = prepare_data(properties_df, data['PKM2_inhibition'])
X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = prepare_data(properties_df, data['ERK2_inhibition'])

rf_PKM2 = train_rf_model(X_train_PKM2, y_train_PKM2,)
rf_ERK2 = train_rf_model(X_train_ERK2, y_train_ERK2)


# Predict on the test set
y_pred_PKM2_rf = rf_PKM2.predict(X_test_PKM2)
y_pred_ERK2_rf = rf_ERK2.predict(X_test_ERK2)

# Calculate the accuracy for PKM2_inhibition
accuracy_PKM2_rf = accuracy_score(y_test_PKM2, y_pred_PKM2_rf)

# Calculate the accuracy for ERK2_inhibition
accuracy_ERK2_rf = accuracy_score(y_test_ERK2, y_pred_ERK2_rf)

# Print the accuracies
print("Random Forest Accuracy for PKM2_inhibition:", accuracy_PKM2_rf)
print("Random Forest Accuracy for ERK2_inhibition:", accuracy_ERK2_rf)

# Compute the confusion matrix for PKM2_inhibition
cm_PKM2_rf = confusion_matrix(y_test_PKM2, y_pred_PKM2_rf)
print("Confusion Matrix for PKM2_inhibition:")
print(cm_PKM2_rf)

# Compute the confusion matrix for ERK2_inhibition
cm_ERK2_rf = confusion_matrix(y_test_ERK2, y_pred_ERK2_rf)
print("Confusion Matrix for ERK2_inhibition:")
print(cm_ERK2_rf)

# Count the values in the predictions
count_PKM2_rf = pd.Series(y_pred_PKM2_rf).value_counts()
count_ERK2_rf = pd.Series(y_pred_ERK2_rf).value_counts()

# Print the count of predictions for PKM2_inhibition
print("Count of predictions for PKM2_inhibition:")
print(count_PKM2_rf)

# Print the count of predictions for ERK2_inhibition
print("Count of predictions for ERK2_inhibition:")
print(count_ERK2_rf)

# Load the untested dataset
untested_data = pd.read_csv("untested_molecules.csv")

# Compute the features for the untested molecules
untested_properties_df = untested_data['SMILES'].apply(lambda x: pd.Series(compute_properties(x)))

# Predict on the untested set using the trained models
untested_pred_PKM2_rf = rf_PKM2.predict(untested_properties_df)
untested_pred_ERK2_rf = rf_ERK2.predict(untested_properties_df)

# Print the predictions for PKM2_inhibition
print("Predictions for PKM2_inhibition:")
print(untested_pred_PKM2_rf)

# Print the predictions for ERK2_inhibition
print("Predictions for ERK2_inhibition:")
print(untested_pred_ERK2_rf)

# Count the values in the predictions for PKM2_inhibition
count_PKM2_pred = pd.Series(untested_pred_PKM2_rf).value_counts()
print("Count of predictions for PKM2_inhibition:")
print(count_PKM2_pred)

# Count the values in the predictions for ERK2_inhibition
count_ERK2_pred = pd.Series(untested_pred_ERK2_rf).value_counts()
print("Count of predictions for ERK2_inhibition:")
print(count_ERK2_pred)


# Determine the most important properties
feature_importances = rf_PKM2.feature_importances_
important_properties = properties_df.columns[feature_importances.argsort()[::-1]]
print("Most important properties for PKM2_inhibition:")
print(important_properties)