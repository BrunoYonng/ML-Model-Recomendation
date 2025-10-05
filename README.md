# Aproveita

App de promoções com recomendação estilo Netflix.

## Estrutura

- frontend/ — React Native (TypeScript)
- backend/ — Node.js + Express + MongoDB
- recommender/ — Flask (Python)

## Requisitos
- Node >= 20, npm >= 10
- Python >= 3.10
- Android SDK (para `npm run android`)

## Inicio Rápido

1. Backend
```bash
npm run backend
```
2. Recommender
```bash
npm run recommender
```
3. Frontend (com emulador Android ligado)
```bash
npm run frontend
```

## Endpoints
- Backend: http://localhost:5000/api/health
- Recommender: http://localhost:5001/health
