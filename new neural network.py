import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv("tested_molecules.csv")

# Function to compute physicochemical properties for PKM2 and ERK2 inhibition
def compute_properties(smiles, target):
    molecule = Chem.MolFromSmiles(smiles)
    properties = {
        'MolLogP': Descriptors.MolLogP(molecule),
        'MolWt': Descriptors.MolWt(molecule),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(molecule),
        'NumHAcceptors': Descriptors.NumHAcceptors(molecule),
        'NumHDonors': Descriptors.NumHDonors(molecule),
        'TPSA': Descriptors.TPSA(molecule),
    }
    
    # Additional properties specific to PKM2 inhibition
    if target == 'PKM2_inhibition':
        properties.update({
            'MinEStateIndex': Descriptors.MinEStateIndex(molecule),
            'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge(molecule),
            'MaxPartialCharge': Descriptors.MaxPartialCharge(molecule),
            'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
            'NumAmideBonds': rdMolDescriptors.CalcNumAmideBonds(molecule),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(molecule),
            'NOCount': Descriptors.NOCount(molecule),
            'Kappa2': rdMolDescriptors.CalcKappa2(molecule),
            'FractionCSP3': Descriptors.FractionCSP3(molecule),
        })
    
    # Additional properties specific to ERK2 inhibition
    elif target == 'ERK2_inhibition':
        properties.update({
            'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(molecule),
            'Chi0v': rdMolDescriptors.CalcChi0v(molecule),
            'NOCount': Descriptors.NOCount(molecule),
            'NumBridgeheadAtoms': rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
            'Kappa1': rdMolDescriptors.CalcKappa1(molecule),
            'Chi2v': rdMolDescriptors.CalcChi2v(molecule),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(molecule),
            'MinPartialCharge': Descriptors.MinPartialCharge(molecule),
            'RingCount': Descriptors.RingCount(molecule),
            'MinAbsEStateIndex': Descriptors.MinAbsEStateIndex(molecule),
        })

    return properties

# Compute physicochemical properties for all molecules for PKM2 inhibition
properties_PKM2_df = data['SMILES'].apply(lambda x: pd.Series(compute_properties(x, 'PKM2_inhibition')))

# Compute physicochemical properties for all molecules for ERK2 inhibition
properties_ERK2_df = data['SMILES'].apply(lambda x: pd.Series(compute_properties(x, 'ERK2_inhibition')))

# Standardize the features for PKM2 inhibition
scaler_PKM2 = StandardScaler()
combined_features_PKM2 = scaler_PKM2.fit_transform(properties_PKM2_df)

# Standardize the features for ERK2 inhibition
scaler_ERK2 = StandardScaler()
combined_features_ERK2 = scaler_ERK2.fit_transform(properties_ERK2_df)

# Apply SMOTE to handle imbalanced dataset
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

# Prepare the data for PKM2 inhibition
def prepare_data_PKM2(data):
    X = combined_features_PKM2
    y = data['PKM2_inhibition']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test

X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = prepare_data_PKM2(data)

# Prepare the data for ERK2 inhibition
def prepare_data_ERK2(data):
    X = combined_features_ERK2
    y = data['ERK2_inhibition']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test

X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = prepare_data_ERK2(data)

# Define the neural network model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model for PKM2 inhibition
model_PKM2 = build_model(X_train_PKM2.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_PKM2.fit(X_train_PKM2, y_train_PKM2, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Build and train the model for ERK2 inhibition
model_ERK2 = build_model(X_train_ERK2.shape[1])
model_ERK2.fit(X_train_ERK2, y_train_ERK2, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Predict on the test set for PKM2 inhibition
y_pred_PKM2_nn = (model_PKM2.predict(X_test_PKM2) > 0.5).astype(int)

# Predict on the test set for ERK2 inhibition
y_pred_ERK2_nn = (model_ERK2.predict(X_test_ERK2) > 0.5).astype(int)

# Calculate the confusion matrix for PKM2 inhibition
cm_PKM2 = confusion_matrix(y_test_PKM2, y_pred_PKM2_nn)
tn_PKM2, fp_PKM2, fn_PKM2, tp_PKM2 = cm_PKM2.ravel()

# Calculate the confusion matrix for ERK2 inhibition
cm_ERK2 = confusion_matrix(y_test_ERK2, y_pred_ERK2_nn)
tn_ERK2, fp_ERK2, fn_ERK2, tp_ERK2 = cm_ERK2.ravel()

# Calculate the metrics for PKM2 inhibition
accuracy_PKM2_nn = accuracy_score(y_test_PKM2, y_pred_PKM2_nn)
precision_PKM2_nn = precision_score(y_test_PKM2, y_pred_PKM2_nn, zero_division=1)
recall_PKM2_nn = recall_score(y_test_PKM2, y_pred_PKM2_nn)
f1_PKM2_nn = f1_score(y_test_PKM2, y_pred_PKM2_nn)
roc_auc_PKM2_nn = roc_auc_score(y_test_PKM2, y_pred_PKM2_nn)
pr_auc_PKM2_nn = average_precision_score(y_test_PKM2, y_pred_PKM2_nn)

# Calculate the metrics for ERK2 inhibition
accuracy_ERK2_nn = accuracy_score(y_test_ERK2, y_pred_ERK2_nn)
precision_ERK2_nn = precision_score(y_test_ERK2, y_pred_ERK2_nn, zero_division=1)
recall_ERK2_nn = recall_score(y_test_ERK2, y_pred_ERK2_nn)
f1_ERK2_nn = f1_score(y_test_ERK2, y_pred_ERK2_nn)
roc_auc_ERK2_nn = roc_auc_score(y_test_ERK2, y_pred_ERK2_nn)
pr_auc_ERK2_nn = average_precision_score(y_test_ERK2, y_pred_ERK2_nn)

# Print the metrics for PKM2 inhibition
print("Neural Network Metrics for PKM2_inhibition:")
print("Accuracy:", accuracy_PKM2_nn)
print("Precision:", precision_PKM2_nn)
print("Recall:", recall_PKM2_nn)
print("F1 Score:", f1_PKM2_nn)
print("ROC AUC:", roc_auc_PKM2_nn)
print("PR AUC:", pr_auc_PKM2_nn)
print("True Positives (TP):", tp_PKM2)
print("False Positives (FP):", fp_PKM2)
print("True Negatives (TN):", tn_PKM2)
print("False Negatives (FN):", fn_PKM2)

# Print the metrics for ERK2 inhibition
print("Neural Network Metrics for ERK2_inhibition:")
print("Accuracy:", accuracy_ERK2_nn)
print("Precision:", precision_ERK2_nn)
print("Recall:", recall_ERK2_nn)
print("F1 Score:", f1_ERK2_nn)
print("ROC AUC:", roc_auc_ERK2_nn)
print("PR AUC:", pr_auc_ERK2_nn)
print("True Positives (TP):", tp_ERK2)
print("False Positives (FP):", fp_ERK2)
print("True Negatives (TN):", tn_ERK2)
print("False Negatives (FN):", fn_ERK2)

# Load the untested dataset
untested_data = pd.read_csv("untested_molecules.csv")

# Compute features for untested molecules for PKM2 inhibition
untested_properties_PKM2_df = untested_data['SMILES'].apply(lambda x: pd.Series(compute_properties(x, 'PKM2_inhibition')))

# Compute features for untested molecules for ERK2 inhibition
untested_properties_ERK2_df = untested_data['SMILES'].apply(lambda x: pd.Series(compute_properties(x, 'ERK2_inhibition')))

# Standardize the features for untested molecules for PKM2 inhibition
untested_combined_features_PKM2 = scaler_PKM2.transform(untested_properties_PKM2_df)

# Standardize the features for untested molecules for ERK2 inhibition
untested_combined_features_ERK2 = scaler_ERK2.transform(untested_properties_ERK2_df)

# Predict on the untested set for PKM2 inhibition
untested_pred_PKM2_nn = (model_PKM2.predict(untested_combined_features_PKM2) > 0.5).astype(int)

# Predict on the untested set for ERK2 inhibition
untested_pred_ERK2_nn = (model_ERK2.predict(untested_combined_features_ERK2) > 0.5).astype(int)

# Print the predictions for untested molecules for PKM2 inhibition
print("Predictions for untested molecules (PKM2_inhibition):")
print(untested_pred_PKM2_nn)

# Print the predictions for untested molecules for ERK2 inhibition
print("\nPredictions for untested molecules (ERK2_inhibition):")
print(untested_pred_ERK2_nn)

# Count the values in the predictions for untested molecules for PKM2 inhibition
print("\nPrediction counts for PKM2_inhibition:")
print(pd.Series(untested_pred_PKM2_nn.flatten()).value_counts())

# Count the values in the predictions for untested molecules for ERK2 inhibition
print("\nPrediction counts for ERK2_inhibition:")
print(pd.Series(untested_pred_ERK2_nn.flatten()).value_counts())
