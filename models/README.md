# Model Files Directory

This directory should contain the trained CatBoost model file:
- `catboost_bp_model.pkl`

## How to Add Your Model

After training your model in the Jupyter notebook (`train_classification_bp.ipynb`), export it using:

```python
import joblib

# Save the trained CatBoost model
joblib.dump(catboost_model, 'ServiceAI/models/catboost_bp_model.pkl')
```

Or use the provided export script:

```python
from export_model import export_model
export_model(catboost_model)
```

## Model Requirements

- **Format**: Pickled scikit-learn compatible model (`.pkl`)
- **Algorithm**: CatBoost Classifier
- **Input Features**: 31 features (must match feature_extractor.py order)
- **Output Classes**: 5 classes (0-4)

## File Size

Typical CatBoost model size: 1-10 MB depending on training parameters.

## Security Note

Do not commit large model files to Git. Use Git LFS or download models separately for deployment.
