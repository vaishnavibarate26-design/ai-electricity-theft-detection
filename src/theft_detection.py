import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# --- CONFIGURATION ---
DATA_PATH = '../data/sgcc_data.csv'
RANDOM_STATE = 42


def load_and_prepare_sgcc():
    """Load SGCC data, handle missing values, and reduce width for speed."""
    print("=" * 50)
    print("STEP 1: Loading SGCC Dataset")
    print("=" * 50)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File not found at {DATA_PATH}")
        print("Current working directory:", os.getcwd())
        return None, None

    # Load the CSV
    df = pd.read_csv(DATA_PATH)
    print(f"Original shape: {df.shape}")

    # Sample for speed & memory safety
    df = df.sample(n=15000, random_state=RANDOM_STATE)
    print(f"Sampled shape: {df.shape}")

    # Show column names briefly
    print(f"First 3 columns: {list(df.columns[:3])}")
    print(f"Last column: {df.columns[-1]}")

    # Label = last column (FLAG: 0=Normal, 1=Theft)
    y = df.iloc[:, -1]

    # Features = middle columns (drop CONS_NO at start and FLAG at end)
    X = df.iloc[:, 1:-1].copy()

    # Force all features to numeric; non-numeric values (strings) become NaN
    X = X.apply(pd.to_numeric, errors='coerce')

    # Drop columns that are 100% NaN
    X = X.dropna(axis=1, how='all')

    # Fill remaining missing values with 0
    X = X.fillna(0)

    # Convert to float32 to save memory
    X = X.astype(np.float32)

    print(f"After cleaning - X shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution:\n{y.value_counts()}")

    return X, y


def train_model(X, y):
    """Train Random Forest with class balancing."""
    print("\n" + "=" * 50)
    print("STEP 2: Training Random Forest (Balanced)")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("Training... (this will take 1-3 minutes)")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Theft']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Theft'],
                yticklabels=['Normal', 'Theft'])
    plt.title(f'SGCC Confusion Matrix (Accuracy: {acc:.2%})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    os.makedirs('../results', exist_ok=True)
    plt.savefig('../results/sgcc_confusion_matrix.png')
    plt.show()

    # Feature importance plot (top 15 most important days)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(10, 6))
    plt.title("Top 15 Most Important Days for Theft Detection")
    plt.bar(range(15), importances[indices])
    plt.xticks(range(15), [str(i) for i in indices], rotation=45)
    plt.xlabel('Day Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png')
    plt.show()

    return model, acc


if __name__ == "__main__":
    X, y = load_and_prepare_sgcc()
    if X is not None:
        model, acc = train_model(X, y)
        print(f"\n{'='*50}")
        print(f"✅ PROJECT COMPLETE!")
        print(f"Final Accuracy: {acc*100:.2f}%")
        print(f"Check results/ folder for saved plots:")
        print(f"  - sgcc_confusion_matrix.png")
        print(f"  - feature_importance.png")
        print(f"{'='*50}")