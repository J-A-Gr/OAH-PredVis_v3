from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('models/simple_logreg_to_buy.pkl')
# Load product data
products = pd.read_csv('data/final/products_imputed.csv')
# Numeric feature columns (drop identifiers/target)
feature_cols = (
    products
    .select_dtypes(include=['int64', 'float64'])
    .drop(columns=['product_id', 'to_buy'], errors='ignore')
    .columns
    .tolist()
)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Ensure it's an integer
    try:
        prod_id = int(data.get('product_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'product_id must be an integer'}), 400

    row = products[products['product_id'] == prod_id]
    if row.empty:
        return jsonify({'error': f'product_id {prod_id} not found'}), 404

    X = row[feature_cols]
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    return jsonify({
        'product_id': prod_id,
        'to_buy_probability': proba,
        'to_buy_prediction': pred
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
