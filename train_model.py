"""
CNN for rotor pressure signal classification
Includes:
- Stratified K-Fold cross validation
- Class imbalance handling
- F1 and MCC evaluation
"""

# ============================================================
# Imports
# ============================================================
import sys
import keras
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ============================================================
# Reproducibility
# ============================================================

seed = 33
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# ============================================================
# Data loader
# ============================================================

def load_npz_for_tensorflow(path):
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    X_tf = X.reshape((-1, X.shape[1], 1)).astype(np.float32)
    y = y.astype(np.int32)
    return X_tf, y

X, y = load_npz_for_tensorflow("student_task_adl_prepared_dataset.npz")



# ============================================================
# Random guessing baseline F1 (based on class distribution)
# ============================================================
p_positive = np.mean(y)
y_rand = np.random.binomial(1, p_positive, size=len(y))
baseline_f1 = f1_score(y, y_rand)

print("Random guessing baseline F1:", baseline_f1)

# ============================================================
# K-Fold Cross Validation
# ============================================================

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

fold = 1
f1_scores = []
mcc_scores = []

for train_index, val_index in kf.split(X, y):
    print(f"\n========== Fold {fold} ==========")
    
    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Compute class weights
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))
    
    # ============================================================
    # Model definition 
    # ============================================================
    
    model = Sequential([
        Input(shape=x_train.shape[1:]),
        Conv1D(4, 3, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(8, 5, padding='same', activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dropout(0.2),
        Dense(16, activation='relu', name="custom_name"),
        Dropout(0.2),
        Dense(1, activation='sigmoid', name="output")
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.summary()
    
    # ============================================================
    # Callbacks
    # ============================================================
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    rlr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        min_lr=1e-7,
        verbose=1
    )
    
    ckpt = ModelCheckpoint(f'best_model_fold{fold}.keras', monitor='val_loss',
                           save_best_only=True, verbose=1)
    
    # ============================================================
    # Training
    # ============================================================
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping, rlr, ckpt],
        verbose=1
    )
    # ============================================================
    # Plot training history
    # ============================================================

    plt.figure()
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel('epochs')
    plt.ylabel('Binary Crossentropy Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"training_loss_fold{fold}.png")
    plt.close()
    
    # ============================================================
    # Evaluation on validation fold
    # ============================================================
    best_model = tf.keras.models.load_model(f"best_model_fold{fold}.keras")  
    y_prob = best_model.predict(x_val).flatten()
    
    threshold = 0.5
    y_pred = (y_prob > threshold).astype(int)
    
    f1 = f1_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    
    print(f"Fold {fold} - F1: {f1:.4f}, MCC: {mcc:.4f}")
    print("Confusion Matrix:\n", cm)
    
    f1_scores.append(f1)
    mcc_scores.append(mcc)
    
    fold += 1

# ============================================================
# Summary of K-Fold results
# ============================================================

print("\n========== K-Fold Summary ==========")
print(f"Mean F1: {np.mean(f1_scores):.4f} \u00B1 {np.std(f1_scores):.4f}")
print(f"Mean MCC: {np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}")

# ============================================================
# Final model
# ============================================================

final_model = Sequential([
    Input(shape=(1400, 1)),
    Conv1D(4, 3, padding='same', activation='relu'),
    MaxPooling1D(2),
    Conv1D(8, 5, padding='same', activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dropout(0.2),
    Dense(16, activation='relu', name="custom_name"),
    Dropout(0.2),
    Dense(1, activation='sigmoid', name="output")
])

final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
final_model.summary()
stop_callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

final_model.fit(
    X, y, 
    validation_split=0.1, 
    epochs=50, 
    batch_size=32, 
    class_weight=class_weights, 
    callbacks=[stop_callback, rlr]
)
 
# save the final model
final_model.save("cnn_rotor_signal_classification.keras")



