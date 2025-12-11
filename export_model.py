"""
Export trained CatBoost model for deployment
Run this script after training to save the model for the service
"""

import joblib
import os

def export_model(model, output_path='../ServiceAI/models/catboost_bp_model.pkl'):
    """
    Export trained model for deployment.
    
    Args:
        model: Trained CatBoost model
        output_path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path)
    print(f"✓ Model exported successfully to: {output_path}")
    
    # Check file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"✓ Model file size: {file_size:.2f} MB")
    
    return output_path

# Example usage in training notebook:
# After training your CatBoost model:
#
# from export_model import export_model
# export_model(catboost_model)
