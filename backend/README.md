# Backend - Aproveita API

## Descrição

API REST em Node.js com Express para gerenciar produtos, supermercados e promoções.

## Tecnologias

- Node.js 20+
- Express
- MongoDB + Mongoose
- Axios (para comunicação com ML API)

## Instalação

```bash
npm install
```

## Configuração

Crie um arquivo `.env`:

```env
PORT=5000
MONGO_URI=mongodb://localhost:27017/aproveita
RECOMMENDER_API=http://127.0.0.1:5001
```

## Executar

```bash
# Desenvolvimento
npm run dev

# Produção
npm start
```

## Endpoints

### Health Check
```http
GET /api/health
```

### Recomendações
```http
GET /api/recommendations
```

### Produtos
```http
GET /api/products
POST /api/products
```

### Supermercados
```http
GET /api/markets
```

### Promoções
```http
GET /api/promotions
```

## Estrutura

```
backend/
├── server.js      # Servidor principal
├── .env          # Variáveis de ambiente
└── package.json
```

## Próximos passos

- [ ] Implementar models do MongoDB
- [ ] Adicionar autenticação JWT
- [ ] Criar routes separadas
- [ ] Adicionar validação de dados
- [ ] Implementar testes
