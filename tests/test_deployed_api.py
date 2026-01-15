"""Test the deployed API against known database values."""
import sqlite3
import requests
import numpy as np
from sklearn.metrics import roc_auc_score

API_URL = "http://136.111.102.10:6515"

def test_deployed_model():
    # Get test samples - use InChIs that have many properties
    conn = sqlite3.connect('brick/cvae.sqlite')
    cursor = conn.execute("""
        SELECT a.inchi, a.property_token, a.value, a.smiles
        FROM activity a
        WHERE a.smiles IS NOT NULL AND a.smiles != ''
        ORDER BY RANDOM()
        LIMIT 100
    """)
    samples = cursor.fetchall()
    conn.close()

    print(f"Testing {len(samples)} samples against deployed API at {API_URL}")

    y_true = []
    y_pred = []
    errors = 0
    success = 0

    for i, (inchi, prop_token, value, smiles) in enumerate(samples):
        try:
            resp = requests.get(
                f"{API_URL}/predict",
                params={"inchi": inchi, "property_token": int(prop_token)},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                pred = data.get("value")
                if pred is not None:
                    y_true.append(value)
                    y_pred.append(pred)
                    success += 1
            else:
                errors += 1
        except Exception as e:
            errors += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples ({success} success, {errors} errors)")

    print(f"\n{'='*50}")
    print("RESULTS:")
    print(f"  Successful predictions: {len(y_true)}")
    print(f"  Errors: {errors}")

    if len(y_true) > 10 and len(set(y_true)) > 1:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        auc = roc_auc_score(y_true, y_pred)
        print(f"  AUC: {auc:.4f}")

        if sum(y_true == 1) > 0:
            print(f"  Mean pred (true=1): {y_pred[y_true == 1].mean():.4f}")
        if sum(y_true == 0) > 0:
            print(f"  Mean pred (true=0): {y_pred[y_true == 0].mean():.4f}")

        accuracy = ((y_pred > 0.5) == y_true).mean()
        print(f"  Accuracy: {accuracy:.4f}")

        if auc > 0.75:
            print("\n✓ PASS: AUC > 0.75")
            return True
        elif auc > 0.6:
            print("\n⚠ WARNING: AUC between 0.6-0.75")
            return True
        else:
            print("\n✗ FAIL: AUC < 0.6")
            return False
    else:
        print("  Not enough valid predictions for AUC calculation")
        return len(y_true) > 5


if __name__ == "__main__":
    import sys
    success = test_deployed_model()
    sys.exit(0 if success else 1)
