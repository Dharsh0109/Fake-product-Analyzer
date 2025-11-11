# Fake or Real Product Analyzer

A web application that uses Random Forest machine learning algorithm to predict whether a product is real or fake based on various product features.

## Features

- **Modern Web Interface**: Clean, responsive design with gradient backgrounds
- **Real-time Prediction**: Instant analysis using trained Random Forest model
- **Percentage Confidence**: Shows probability percentages for real/fake classification
- **Auto-calculation**: Automatically calculates discount percentage
- **Error Handling**: Comprehensive error handling and user feedback

## Files Structure

```
fake or real product analyzer/
├── app.py                          # Flask web application
├── templates/
│   └── index.html                  # Web interface template
├── fake_or_real.ipynb             # Jupyter notebook with model training
├── fake_product_analyzer.pkl      # Trained model (generated)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Open Browser**:
   Navigate to `http://localhost:5000`

## How to Use

1. **Enter Product Details**:
   - Brand Name (Code): Numeric code for brand (0-10)
   - Product ID: Unique product identifier (1000-9999)
   - Product Name (Code): Numeric code for product name
   - Brand Description (Code): Numeric code for brand description
   - Product Size (Code): Numeric code for product size
   - Currency: Select currency type
   - MRP: Maximum Retail Price
   - Selling Price: Current selling price
   - Discount: Discount percentage (auto-calculated)
   - Category: Product category

2. **Click "Analyze Product Authenticity"**

3. **View Results**:
   - Green result: Product is likely **Real**
   - Red result: Product is likely **Fake**
   - Percentage confidence is displayed

## Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 10 product attributes
- **Training**: Based on price ratios and product characteristics
- **Accuracy**: High accuracy on test data (see notebook for details)

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **ML Library**: scikit-learn
- **Model Persistence**: joblib

## Sample Data

The application comes with sample default values that represent a real product:
- Brand: 2, Product ID: 4564, MRP: 3900, Selling Price: 3120, etc.

## Notes

- The model uses encoded categorical variables
- Price ratio (Selling Price vs MRP) is a key factor in classification
- All numeric inputs are validated for appropriate ranges