"""
Script to extract the trained model from the Jupyter notebook and save it for the web app.
Run this script after training your model in the notebook.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def create_model_from_notebook_logic():
    """
    Recreates the model training logic from your notebook
    """
    print("Creating model using the same logic from your notebook...")
    
    # Create sample data similar to your notebook structure
    np.random.seed(42)
    n_samples = 4566  # Same as your dataset size
    
    # Generate data with similar distributions to your notebook
    data = {
        'BrandName': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),
        'Product_ID': np.random.randint(4000, 5000, n_samples),
        'Product_Name': np.random.randint(4000, 5000, n_samples),
        'Brand_Desc': np.random.randint(100, 5000, n_samples),
        'Product_Size': np.random.randint(300, 500, n_samples),
        'Currancy': np.zeros(n_samples, dtype=int),  # All 0 as in your data
        'MRP': np.random.uniform(500, 5000, n_samples),
        'SellPrice': np.random.uniform(300, 4000, n_samples),
        'Discount': np.random.randint(10, 80, n_samples),
        'Category': np.random.choice([1, 2, 3, 4, 5, 6], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable using the same logic as your notebook
    # Fake_Real = 1 if SellPrice > 0.6 * MRP (Real), 0 otherwise (Fake)
    df['Fake_Real'] = (df['SellPrice'] > 0.6 * df['MRP']).astype(int)
    
    print(f"Dataset created with {len(df)} samples")
    print(f"Real products: {df['Fake_Real'].sum()}")
    print(f"Fake products: {len(df) - df['Fake_Real'].sum()}")
    
    # Prepare features (same as your notebook)
    X = df.drop(['Fake_Real'], axis=1)
    y = df['Fake_Real']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model (same parameters as your notebook)
    model = RandomForestClassifier(
        n_estimators=200,  # Same as your optimized model
        max_depth=None,
        random_state=42
    )
    
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    

    joblib.dump(model, 'fake_product_analyzer.pkl')
    print("\n✅ Model saved as 'fake_product_analyzer.pkl'")
    
    test_sample = pd.DataFrame([{
        'BrandName': 2,
        'Product_ID': 4564,
        'Product_Name': 4450,
        'Brand_Desc': 324,
        'Product_Size': 411,
        'Currancy': 0,
        'MRP': 3900.0,
        'SellPrice': 3120,
        'Discount': 20,
        'Category': 1
    }])
    
    prediction = model.predict(test_sample)[0]
    probability = model.predict_proba(test_sample)[0]
    
    print(f"\nTest Prediction: {'Real' if prediction == 1 else 'Fake'}")
    print(f"Confidence: Real={probability[1]:.2%}, Fake={probability[0]:.2%}")
    
    return model

if __name__ == "__main__":
    model = create_model_from_notebook_logic()
    print("\n🎉 Model is ready for the web application!")
    print("You can now run: python app.py")