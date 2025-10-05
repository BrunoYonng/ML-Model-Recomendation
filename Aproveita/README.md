# Aproveita Monorepo

Este repositório contém três serviços:

- frontend: App React Native (CLI) estilo Netflix
- backend: API Node.js/Express com MongoDB
- recommender: API Flask para recomendações

## Comandos rápidos

- Backend: `cd backend && npm run dev`
- Recommender: `cd recommender && . venv/bin/activate && python app.py`
- Frontend (Android): `cd frontend && npm run android`

## Variáveis de ambiente

Edite o arquivo `.env` na raiz:

```
PORT=5000
MONGO_URI=mongodb://localhost:27017/aproveita
RECOMMENDER_API=http://127.0.0.1:5001
```

## Endpoints de saúde

- Backend: http://localhost:5000/api/health
- Recommender: http://localhost:5001/health
