import matplotlib.pyplot as plt
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
    """
    Compute the properties of a molecule based on its SMILES string using RDKit.
    
    Args:
    - smiles (str): SMILES string of the molecule.
    - target (str): Target for inhibition prediction ('PKM2_inhibition' or 'ERK2_inhibition').
    
    Returns:
    - properties (dict): Dictionary containing computed properties.
    """
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
    
# Function to compute physicochemical properties for PKM2 and ERK2 inhibition
def compute_properties(smiles, target):
    """
    Compute the properties of a molecule based on its SMILES string using RDKit.
    
    Args:
    - smiles (str): SMILES string of the molecule.
    - target (str): Target for inhibition prediction ('PKM2_inhibition' or 'ERK2_inhibition').
    
    Returns:
    - properties (dict): Dictionary containing computed properties.
    """
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
        # Add PKM2 specific properties here
        pass

    return properties

# Apply SMOTE to handle imbalanced dataset
def apply_smote(X, y):
    """
    Apply Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset before training the model.
    
    Args:
    - X (array): Feature matrix
    - y (array): Target labels
    
    Returns:
    - X_resampled (array): Resampled feature matrix.
    - y_resampled (array): Resampled target labels.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

# Prepare the data for PKM2 inhibition
def prepare_data_PKM2(data):
    """
    Prepare data for PKM2 inhibition.

    Args:
    - data (DataFrame): DataFrame containing molecule SMILES and 'PKM2_inhibition' label.

    Returns:
    - X_train (array): Resampled training features for PKM2 inhibition.
    - X_test (array): Testing features for PKM2 inhibition.
    - y_train (array): Resampled training labels for PKM2 inhibition.
    - y_test (array): Testing labels for PKM2 inhibition.
    """
    X = combined_features_PKM2
    y = data['PKM2_inhibition']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test

# Prepare the data for ERK2 inhibition
def prepare_data_ERK2(data):
    """
    Prepare data for ERK2 inhibition.

    Args:
    - data (DataFrame): DataFrame containing molecule SMILES and 'ERK2_inhibition' label.

    Returns:
    - X_train (array): Resampled training features for ERK2 inhibition.
    - X_test (array): Testing features for ERK2 inhibition.
    - y_train (array): Resampled training labels for ERK2 inhibition.
    - y_test (array): Testing labels for ERK2 inhibition.
    """
    X = combined_features_ERK2
    y = data['ERK2_inhibition']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    return X_train_resampled, X_test, y_train_resampled, y_test

# Define the neural network model
def build_model(input_dim):
    """
    Build a neural network model for inhibition prediction.

    Args:
    - input_dim (int): Dimension of the input features.

    Returns:
    - model (Sequential): Compiled neural network model.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix.

    Args:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.

    Returns:
    - tn (int): True negatives.
    - fp (int): False positives.
    - fn (int): False negatives.
    - tp (int): True positives.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

def calculate_metrics(y_true, y_pred):
    """
    Calculate the evaluation metrics.

    Args:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.

    Returns:
    - accuracy (float): Accuracy.
    - precision (float): Precision.
    - recall (float): Recall.
    - f1_score (float): F1 score.
    - roc_auc (float): ROC AUC score.
    - pr_auc (float): PR AUC score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc, pr_auc

def plot_metrics(history_PKM2, history_ERK2):
    """
    Plot the accuracy and loss metrics for PKM2 and ERK2 inhibition.

    Args:
    - history_PKM2 (History): Training history for PKM2 inhibition.
    - history_ERK2 (History): Training history for ERK2 inhibition.
    """
    import matplotlib.pyplot as plt

    # Plot the accuracy and val_loss for PKM2 inhibition
    plt.figure(figsize=(10, 6))
    plt.plot(history_PKM2.history['accuracy'], label='Train Accuracy')
    plt.plot(history_PKM2.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history_PKM2.history['loss'], label='Train Loss')
    plt.plot(history_PKM2.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Metrics for PKM2 Inhibition')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()

    # Plot the accuracy and val_loss for ERK2 inhibition
    plt.figure(figsize=(10, 6))
    plt.plot(history_ERK2.history['accuracy'], label='Train Accuracy')
    plt.plot(history_ERK2.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history_ERK2.history['loss'], label='Train Loss')
    plt.plot(history_ERK2.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Metrics for ERK2 Inhibition')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()

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

# Prepare the data for PKM2 inhibition
X_train_PKM2, X_test_PKM2, y_train_PKM2, y_test_PKM2 = prepare_data_PKM2(data)

# Prepare the data for ERK2 inhibition
X_train_ERK2, X_test_ERK2, y_train_ERK2, y_test_ERK2 = prepare_data_ERK2(data)

# Build and train the model for PKM2 inhibition
model_PKM2 = build_model(X_train_PKM2.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history_PKM2 = model_PKM2.fit(X_train_PKM2, y_train_PKM2, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Build and train the model for ERK2 inhibition
model_ERK2 = build_model(X_train_ERK2.shape[1])
history_ERK2 = model_ERK2.fit(X_train_ERK2, y_train_ERK2, epochs=30, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Predict on the test set for PKM2 inhibition
y_pred_PKM2_nn = (model_PKM2.predict(X_test_PKM2) > 0.5).astype(int)
print(y_pred_PKM2_nn)

# Predict on the test set for ERK2 inhibition
y_pred_ERK2_nn = (model_ERK2.predict(X_test_ERK2) > 0.5).astype(int)

# Calculate the confusion matrix for PKM2 inhibition
tn_PKM2, fp_PKM2, fn_PKM2, tp_PKM2 = calculate_confusion_matrix(y_test_PKM2, y_pred_PKM2_nn)

# Calculate the confusion matrix for ERK2 inhibition
tn_ERK2, fp_ERK2, fn_ERK2, tp_ERK2 = calculate_confusion_matrix(y_test_ERK2, y_pred_ERK2_nn)

# Calculate the metrics for PKM2 inhibition
accuracy_PKM2_nn, precision_PKM2_nn, recall_PKM2_nn, f1_PKM2_nn, roc_auc_PKM2_nn, pr_auc_PKM2_nn = calculate_metrics(y_test_PKM2, y_pred_PKM2_nn)

# Calculate the metrics for ERK2 inhibition
accuracy_ERK2_nn, precision_ERK2_nn, recall_ERK2_nn, f1_ERK2_nn, roc_auc_ERK2_nn, pr_auc_ERK2_nn = calculate_metrics(y_test_ERK2, y_pred_ERK2_nn)

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

# Plot the accuracy and loss metrics for PKM2 and ERK2 inhibition
plot_metrics(history_PKM2, history_ERK2)

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

# Save the predictions to the untested data file
untested_data['PKM2_inhibition'] = untested_pred_PKM2_nn
untested_data['ERK2_inhibition'] = untested_pred_ERK2_nn
untested_data.to_csv("untested_molecules_predictions.csv", index=False)