from flask import Flask, render_template, request, jsonify
import numpy as np
import hashlib

app = Flask(__name__)

def analyze_product(product_name, brand_name, price):
    product_lower = product_name.lower()
    brand_lower = brand_name.lower()
    

    fake_keywords = ['replica', 'copy', 'fake', 'imitation', 'knockoff', 'cheap', 'wholesale']
    suspicious_patterns = ['v1', 'v2', 'aaa', '1:1', 'mirror']
    
    fake_score = 0

    for keyword in fake_keywords:
        if keyword in product_lower or keyword in brand_lower:
            fake_score += 30
    
    for pattern in suspicious_patterns:
        if pattern in product_lower:
            fake_score += 25
    
    if price < 4000:  
        fake_score += 20
    elif price < 8000:  
        fake_score += 10
    
    known_brands = ['apple', 'samsung', 'nike', 'adidas', 'gucci', 'louis vuitton', 'rolex']
    if any(brand in brand_lower for brand in known_brands):
        if price < 16000: 
            fake_score += 40
    

    hash_input = f"{product_name}{brand_name}{price}"
    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
    random_factor = (hash_value % 20) - 10 
    
    fake_score += random_factor
    fake_score = max(0, min(100, fake_score)) 
    
    return fake_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        product_name = request.form['product_name']
        brand_name = request.form['brand_name']
        price = float(request.form['price'])
        

        fake_percentage = analyze_product(product_name, brand_name, price)
        real_percentage = 100 - fake_percentage
        
        result = {
            'fake_percentage': round(fake_percentage, 1),
            'real_percentage': round(real_percentage, 1)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)