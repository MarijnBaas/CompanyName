import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('tested_molecules.csv')

# Function to convert SMILES to different fingerprints
def smiles_to_fingerprints(smiles, radius=2, nBits=1024):
    molecule = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits)

# Function to compute physicochemical properties
def compute_properties(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    properties = {
        'MolWt': Descriptors.MolWt(molecule),
        'MolLogP': Descriptors.MolLogP(molecule),
        'NumHDonors': Descriptors.NumHDonors(molecule),
        'NumHAcceptors': Descriptors.NumHAcceptors(molecule),
        'TPSA': Descriptors.TPSA(molecule)
    }
    return properties

# Function to compute pharmacophore features
def compute_pharmacophore(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    pharmacophore = {
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(molecule),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(molecule)
    }
    return pharmacophore

# Apply the functions to the data
data['Fingerprints'] = data['SMILES'].apply(lambda x: smiles_to_fingerprints(x).ToBitString())
data = data.dropna(subset=['Fingerprints'])

fingerprints_df = data['Fingerprints'].apply(lambda x: pd.Series(list(map(int, x))))
fingerprints_df.columns = [f'Bit_{i}' for i in range(fingerprints_df.shape[1])]

properties_df = data['SMILES'].apply(lambda x: pd.Series(compute_properties(x)))
pharmacophore_df = data['SMILES'].apply(lambda x: pd.Series(compute_pharmacophore(x)))

# Combine all features
combined_features_df = pd.concat([fingerprints_df, properties_df, pharmacophore_df], axis=1)

def prepare_data(data, target):
    # Prepare feature and target variables
    X = data
    y = target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_rf_model(X_train, y_train):
    # Train a Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = prepare_data(combined_features_df, data['PKM2_inhibition'])
X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = prepare_data(combined_features_df, data['ERK2_inhibition'])

rf_PKM2 = train_rf_model(X_train_PKM2, y_train_PKM2)
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
