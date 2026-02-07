"""
Train BBB Classifier on Real Dataset.

This script trains the BBB permeability classifier and saves the model checkpoint.
Based on notebooks/01_bbb_classifier_training_REAL.ipynb but automated.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import pickle

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available, skipping plots")

# Random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class BBBClassifier(nn.Module):
    """MLP classifier for BBB permeability prediction."""

    def __init__(self, input_dim=9, hidden_dim1=64, hidden_dim2=32, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_classifier(
    data_path='data/bbb/bbb_dataset_real_with_features.csv',
    output_dir='models',
    n_epochs=100,
    batch_size=32,
    lr=0.001,
    weight_decay=1e-4,
    early_stop_patience=20,
):
    """Train BBB classifier."""

    print("=" * 80)
    print("BBB CLASSIFIER TRAINING")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"   Total: {len(df)} peptides")
    print(f"   BBB+:  {(df['label'] == 1).sum()}")
    print(f"   BBB-:  {(df['label'] == 0).sum()}")

    # Extract features and labels
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    print(f"\n2. Using {len(feature_cols)} features")

    X = df[feature_cols].values
    y = df['label'].values
    splits = df['split'].values

    X_train = X[splits == 'train']
    y_train = y[splits == 'train']
    X_val = X[splits == 'val']
    y_val = y[splits == 'val']
    X_test = X[splits == 'test']
    y_test = y[splits == 'test']

    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")

    # Normalize features
    print("\n3. Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_scaled)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    # Compute class weights
    print("\n4. Computing class weights...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_train
    )
    pos_weight = torch.FloatTensor([class_weights[1] / class_weights[0]])
    print(f"   Class weights: BBB-={class_weights[0]:.3f}, BBB+={class_weights[1]:.3f}")
    print(f"   pos_weight:    {pos_weight.item():.2f}")

    # Initialize model
    print("\n5. Initializing model...")
    input_dim = X_train_scaled.shape[1]
    model = BBBClassifier(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32, dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,}")

    # Training setup
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    print(f"\n6. Training for up to {n_epochs} epochs...")
    print("=" * 80)

    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_probs = torch.sigmoid(val_logits).numpy().flatten()
            val_preds = (val_probs > 0.5).astype(int)

            val_auc = roc_auc_score(y_val, val_probs)
            val_f1 = f1_score(y_val, val_preds)

            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_f1'].append(val_f1)

        scheduler.step(val_loss)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auc:.4f} | "
                  f"Val F1: {val_f1:.4f} | Best: {best_val_auc:.4f}")

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Test evaluation
    print("\n" + "=" * 80)
    print("7. Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_probs = torch.sigmoid(test_logits).numpy().flatten()
        test_preds = (test_probs > 0.5).astype(int)

    test_auc = roc_auc_score(y_test, test_probs)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds, zero_division=0)
    test_recall = recall_score(y_test, test_preds)

    print(f"\nTest Set Performance:")
    print(f"  AUROC:     {test_auc:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")

    # Save model
    print("\n8. Saving model...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / 'bbb_classifier.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim1': 64,
        'hidden_dim2': 32,
        'dropout': 0.3,
        'feature_cols': feature_cols,
        'test_auroc': test_auc,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'class_weights': class_weights,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
    }, model_path)

    scaler_path = output_dir / 'bbb_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"   Model saved to: {model_path}")
    print(f"   Scaler saved to: {scaler_path}")

    # Save training curves
    if HAS_PLOTTING:
        print("\n9. Saving plots...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(history['train_loss'], label='Train', alpha=0.7)
        axes[0].plot(history['val_loss'], label='Val', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(history['val_auc'], color='green', alpha=0.7)
        axes[1].axhline(y=best_val_auc, color='blue', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUROC')
        axes[1].set_title(f'Validation AUROC (Best: {best_val_auc:.4f})')
        axes[1].grid(alpha=0.3)

        axes[2].plot(history['val_f1'], color='orange', alpha=0.7)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Validation F1')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'bbb_training_curves.png', dpi=150, bbox_inches='tight')
        print(f"   Plots saved to: {output_dir / 'bbb_training_curves.png'}")
    else:
        print("\n9. Skipping plots (matplotlib not available)")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Test Performance:")
    print(f"  AUROC: {test_auc:.4f}")
    print(f"  F1:    {test_f1:.4f}")
    print(f"  Acc:   {test_acc:.4f}")

    return model, scaler, history


if __name__ == "__main__":
    model, scaler, history = train_classifier()
