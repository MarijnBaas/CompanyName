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

# Function to compute physicochemical properties
def compute_properties(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    properties = {
        'Ipc': Descriptors.Ipc(molecule),
        'BertzCT': Descriptors.BertzCT(molecule),
        'MolLogP': Descriptors.MolLogP(molecule),
        'Ipc': Descriptors.Ipc(molecule),
        'MolWt': Descriptors.MolWt(molecule),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(molecule),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(molecule),
        'NumHAcceptors': Descriptors.NumHAcceptors(molecule),
        'NumHDonors': Descriptors.NumHDonors(molecule),
    }
    return properties

# Function to compute pharmacophore features
def compute_pharmacophore(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    pharmacophore = {
        'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(molecule),
        'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(molecule),
    }
    return pharmacophore

properties_df = data['SMILES'].apply(lambda x: pd.Series(compute_properties(x)))
pharmacophore_df = data['SMILES'].apply(lambda x: pd.Series(compute_pharmacophore(x)))

# Combine all features
combined_features_df = pd.concat([properties_df, pharmacophore_df], axis=1)

# Standardize the features
scaler = StandardScaler()
combined_features_df = scaler.fit_transform(combined_features_df)

# Apply SMOTE to handle imbalanced dataset
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

# Prepare the data
def prepare_data(data, target):
    X = data
    y = target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test

X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = prepare_data(combined_features_df, data['PKM2_inhibition'])
X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = prepare_data(combined_features_df, data['ERK2_inhibition'])

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

# Build and train the model for PKM2
model_PKM2 = build_model(X_train_PKM2.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_PKM2.fit(X_train_PKM2, y_train_PKM2, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Build and train the model for ERK2
model_ERK2 = build_model(X_train_ERK2.shape[1])
model_ERK2.fit(X_train_ERK2, y_train_ERK2, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Predict on the test set
y_pred_PKM2_nn = (model_PKM2.predict(X_test_PKM2) > 0.5).astype(int)
y_pred_ERK2_nn = (model_ERK2.predict(X_test_ERK2) > 0.5).astype(int)

# Calculate the confusion matrix for PKM2
cm_PKM2 = confusion_matrix(y_test_PKM2, y_pred_PKM2_nn)
tn_PKM2, fp_PKM2, fn_PKM2, tp_PKM2 = cm_PKM2.ravel()

# Calculate the confusion matrix for ERK2
cm_ERK2 = confusion_matrix(y_test_ERK2, y_pred_ERK2_nn)
tn_ERK2, fp_ERK2, fn_ERK2, tp_ERK2 = cm_ERK2.ravel()

# Calculate the metrics for PKM2_inhibition
accuracy_PKM2_nn = accuracy_score(y_test_PKM2, y_pred_PKM2_nn)
precision_PKM2_nn = precision_score(y_test_PKM2, y_pred_PKM2_nn, zero_division=1)
recall_PKM2_nn = recall_score(y_test_PKM2, y_pred_PKM2_nn)
f1_PKM2_nn = f1_score(y_test_PKM2, y_pred_PKM2_nn)
roc_auc_PKM2_nn = roc_auc_score(y_test_PKM2, y_pred_PKM2_nn)
pr_auc_PKM2_nn = average_precision_score(y_test_PKM2, y_pred_PKM2_nn)

# Calculate the metrics for ERK2_inhibition
accuracy_ERK2_nn = accuracy_score(y_test_ERK2, y_pred_ERK2_nn)
precision_ERK2_nn = precision_score(y_test_ERK2, y_pred_ERK2_nn, zero_division=1)
recall_ERK2_nn = recall_score(y_test_ERK2, y_pred_ERK2_nn)
f1_ERK2_nn = f1_score(y_test_ERK2, y_pred_ERK2_nn)
roc_auc_ERK2_nn = roc_auc_score(y_test_ERK2, y_pred_ERK2_nn)
pr_auc_ERK2_nn = average_precision_score(y_test_ERK2, y_pred_ERK2_nn)

# Print the metrics
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

print("\nNeural Network Metrics for ERK2_inhibition:")
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