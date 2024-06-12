import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data = pd.read_csv('tested_molecules.csv')

# Function to convert SMILES to different fingerprints
def smiles_to_fingerprints(smiles, radius=2, nBits=1024):
    molecule = Chem.MolFromSmiles(smiles)
    
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits)

data['Fingerprints'] = data['SMILES'].apply(lambda x: smiles_to_fingerprints(x).ToBitString())

# Filter out any molecules that could not be processed
data = data.dropna(subset=['Fingerprints'])

# Convert fingerprints to a DataFrame
fingerprints_df = data['Fingerprints'].apply(lambda x: pd.Series(list(map(int, x))))
fingerprints_df.columns = [f'Bit_{i}' for i in range(fingerprints_df.shape[1])]

def prepare_data(data, target):
    # Prepare feature and target variables
    X = data
    y = target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Train a Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    return rf

X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = prepare_data(fingerprints_df, data['PKM2_inhibition'])
X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = prepare_data(fingerprints_df, data['ERK2_inhibition'])

rf_PKM2 = train_model(X_train_PKM2, y_train_PKM2)
rf_ERK2 = train_model(X_train_ERK2, y_train_ERK2)

# Predict on the test set
y_pred_PKM2 = rf_PKM2.predict(X_test_PKM2)
y_pred_ERK2 = rf_ERK2.predict(X_test_ERK2)


# Calculate the number of correct predictions for PKM2_inhibition
correct_predictions_PKM2 = sum(y_pred_PKM2 == y_test_PKM2)

# Calculate the number of correct predictions for ERK2_inhibition
correct_predictions_ERK2 = sum(y_pred_ERK2 == y_test_ERK2)

# Print the number of correct predictions
print("Number of correct predictions for PKM2_inhibition:", correct_predictions_PKM2)
print("Number of correct predictions for ERK2_inhibition:", correct_predictions_ERK2)
