from flask import Flask, jsonify, request
import random

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'service': 'Aproveita ML Recommender',
        'version': '1.0.0'
    })

@app.route('/recommend', methods=['GET'])
def recommend():
    """
    Endpoint para recomendações de produtos
    Futuramente irá usar ML para personalizar recomendações
    """
    # Produtos de exemplo
    all_products = [
        {'id': 1, 'name': 'Banana', 'category': 'Frutas', 'discount': 20},
        {'id': 2, 'name': 'Tomate', 'category': 'Legumes', 'discount': 15},
        {'id': 3, 'name': 'Leite', 'category': 'Laticínios', 'discount': 10},
        {'id': 4, 'name': 'Arroz', 'category': 'Grãos', 'discount': 25},
        {'id': 5, 'name': 'Maçã', 'category': 'Frutas', 'discount': 18},
        {'id': 6, 'name': 'Cebola', 'category': 'Legumes', 'discount': 12},
        {'id': 7, 'name': 'Iogurte', 'category': 'Laticínios', 'discount': 30},
        {'id': 8, 'name': 'Feijão', 'category': 'Grãos', 'discount': 22}
    ]
    
    # Selecionar 4 produtos aleatórios para recomendar
    recommendations = random.sample(all_products, min(4, len(all_products)))
    
    return jsonify({
        'recommendations': recommendations,
        'count': len(recommendations),
        'algorithm': 'random_v1'
    }), 200

@app.route('/recommend/user/<int:user_id>', methods=['GET'])
def recommend_for_user(user_id):
    """
    Endpoint para recomendações personalizadas por usuário
    """
    products = [
        {'id': 1, 'name': 'Banana', 'score': 0.95},
        {'id': 2, 'name': 'Tomate', 'score': 0.87},
        {'id': 3, 'name': 'Leite', 'score': 0.82},
        {'id': 4, 'name': 'Arroz', 'score': 0.78}
    ]
    
    return jsonify({
        'user_id': user_id,
        'recommendations': products,
        'personalized': True
    }), 200

@app.route('/recommend/category/<string:category>', methods=['GET'])
def recommend_by_category(category):
    """
    Endpoint para recomendações por categoria
    """
    category_products = {
        'frutas': ['Banana', 'Maçã', 'Laranja', 'Uva'],
        'legumes': ['Tomate', 'Cebola', 'Cenoura', 'Batata'],
        'laticinios': ['Leite', 'Iogurte', 'Queijo', 'Manteiga'],
        'graos': ['Arroz', 'Feijão', 'Lentilha', 'Grão-de-bico']
    }
    
    products = category_products.get(category.lower(), [])
    
    return jsonify({
        'category': category,
        'recommendations': products
    }), 200

if __name__ == '__main__':
    print("🤖 Iniciando serviço de recomendações ML...")
    print("📍 Endpoints disponíveis:")
    print("   - GET /health")
    print("   - GET /recommend")
    print("   - GET /recommend/user/<user_id>")
    print("   - GET /recommend/category/<category>")
    app.run(host='0.0.0.0', port=5001, debug=True)
