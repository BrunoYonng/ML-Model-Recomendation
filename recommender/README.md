# Recommender - ML API

## Descrição

Serviço de recomendações inteligentes usando Flask e Machine Learning.

## Tecnologias

- Python 3.10+
- Flask
- Pandas
- Scikit-learn
- NumPy

## Instalação

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## Executar

```bash
python app.py
```

O serviço estará disponível em `http://localhost:5001`

## Endpoints

### Health Check
```http
GET /health
```

Resposta:
```json
{
  "status": "running",
  "service": "Aproveita ML Recommender",
  "version": "1.0.0"
}
```

### Recomendações Gerais
```http
GET /recommend
```

Resposta:
```json
{
  "recommendations": [
    {
      "id": 1,
      "name": "Banana",
      "category": "Frutas",
      "discount": 20
    }
  ],
  "count": 4,
  "algorithm": "random_v1"
}
```

### Recomendações por Usuário
```http
GET /recommend/user/<user_id>
```

### Recomendações por Categoria
```http
GET /recommend/category/<category>
```

Categorias disponíveis:
- frutas
- legumes
- laticinios
- graos

## Algoritmos de ML

### Versão Atual (v1.0)
- Recomendações aleatórias para demonstração

### Roadmap
- [ ] Collaborative Filtering
- [ ] Content-Based Filtering
- [ ] Hybrid Recommender System
- [ ] Deep Learning (Neural Networks)
- [ ] A/B Testing framework

## Estrutura

```
recommender/
├── app.py              # API Flask
├── requirements.txt    # Dependências
├── venv/              # Ambiente virtual
└── models/            # (futuro) Modelos treinados
```

## Desenvolvimento

Para adicionar novos algoritmos de recomendação:

1. Crie uma nova função em `app.py`
2. Implemente o algoritmo
3. Crie um novo endpoint
4. Documente o endpoint aqui

## Testes

```bash
# Testar localmente
curl http://localhost:5001/health
curl http://localhost:5001/recommend
```
