from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({'status': 'running'})

@app.route('/recommend')
def recommend():
    products = ['Banana', 'Tomate', 'Leite', 'Arroz']
    return jsonify({'recommendations': products}), 200

if __name__ == '__main__':
    print("🤖 Iniciando serviço de recomendações...")
    app.run(host='0.0.0.0', port=5001, debug=True)
