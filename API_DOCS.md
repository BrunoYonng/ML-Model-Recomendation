# 📚 Documentação da API - Aproveita

## Visão Geral

O sistema Aproveita possui duas APIs principais:

1. **Backend API** (Node.js) - Porta 5000
2. **Recommender API** (Flask/Python) - Porta 5001

---

## 🔷 Backend API (Node.js)

**Base URL**: `http://localhost:5000/api`

### Endpoints

#### 1. Health Check

```http
GET /api/health
```

**Descrição**: Verifica o status do servidor backend.

**Resposta de Sucesso** (200 OK):
```json
{
  "status": "ok",
  "message": "Backend Aproveita está rodando!",
  "timestamp": "2025-10-05T10:30:00.000Z"
}
```

**Exemplo**:
```bash
curl http://localhost:5000/api/health
```

---

#### 2. Obter Recomendações

```http
GET /api/recommendations
```

**Descrição**: Busca recomendações do serviço de ML (Flask).

**Resposta de Sucesso** (200 OK):
```json
{
  "recommendations": [
    {
      "id": 1,
      "name": "Banana",
      "category": "Frutas",
      "discount": 20
    },
    {
      "id": 2,
      "name": "Tomate",
      "category": "Legumes",
      "discount": 15
    }
  ],
  "count": 4,
  "algorithm": "random_v1"
}
```

**Resposta de Erro** (500):
```json
{
  "error": "Erro ao buscar recomendações",
  "message": "connect ECONNREFUSED 127.0.0.1:5001"
}
```

**Exemplo**:
```bash
curl http://localhost:5000/api/recommendations
```

---

#### 3. Listar Produtos

```http
GET /api/products
```

**Descrição**: Retorna lista de produtos cadastrados.

**Resposta de Sucesso** (200 OK):
```json
{
  "message": "Lista de produtos",
  "products": []
}
```

**Exemplo**:
```bash
curl http://localhost:5000/api/products
```

---

#### 4. Criar Produto

```http
POST /api/products
```

**Descrição**: Cadastra um novo produto.

**Body**:
```json
{
  "name": "Arroz",
  "category": "Grãos",
  "price": 5.99,
  "discount": 10
}
```

**Resposta de Sucesso** (200 OK):
```json
{
  "message": "Produto criado com sucesso",
  "product": {
    "name": "Arroz",
    "category": "Grãos",
    "price": 5.99,
    "discount": 10
  }
}
```

**Exemplo**:
```bash
curl -X POST http://localhost:5000/api/products \
  -H "Content-Type: application/json" \
  -d '{"name":"Arroz","category":"Grãos","price":5.99,"discount":10}'
```

---

#### 5. Listar Supermercados

```http
GET /api/markets
```

**Descrição**: Retorna lista de supermercados.

**Resposta de Sucesso** (200 OK):
```json
{
  "message": "Lista de supermercados",
  "markets": []
}
```

**Exemplo**:
```bash
curl http://localhost:5000/api/markets
```

---

#### 6. Listar Promoções

```http
GET /api/promotions
```

**Descrição**: Retorna promoções ativas.

**Resposta de Sucesso** (200 OK):
```json
{
  "message": "Lista de promoções",
  "promotions": []
}
```

**Exemplo**:
```bash
curl http://localhost:5000/api/promotions
```

---

## 🔶 Recommender API (Flask)

**Base URL**: `http://localhost:5001`

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Descrição**: Verifica o status do serviço de recomendações.

**Resposta de Sucesso** (200 OK):
```json
{
  "status": "running",
  "service": "Aproveita ML Recommender",
  "version": "1.0.0"
}
```

**Exemplo**:
```bash
curl http://localhost:5001/health
```

---

#### 2. Recomendações Gerais

```http
GET /recommend
```

**Descrição**: Retorna recomendações gerais de produtos.

**Resposta de Sucesso** (200 OK):
```json
{
  "recommendations": [
    {
      "id": 1,
      "name": "Banana",
      "category": "Frutas",
      "discount": 20
    },
    {
      "id": 5,
      "name": "Maçã",
      "category": "Frutas",
      "discount": 18
    },
    {
      "id": 3,
      "name": "Leite",
      "category": "Laticínios",
      "discount": 10
    },
    {
      "id": 8,
      "name": "Feijão",
      "category": "Grãos",
      "discount": 22
    }
  ],
  "count": 4,
  "algorithm": "random_v1"
}
```

**Exemplo**:
```bash
curl http://localhost:5001/recommend
```

---

#### 3. Recomendações por Usuário

```http
GET /recommend/user/<user_id>
```

**Parâmetros**:
- `user_id` (int): ID do usuário

**Descrição**: Retorna recomendações personalizadas para um usuário específico.

**Resposta de Sucesso** (200 OK):
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "id": 1,
      "name": "Banana",
      "score": 0.95
    },
    {
      "id": 2,
      "name": "Tomate",
      "score": 0.87
    },
    {
      "id": 3,
      "name": "Leite",
      "score": 0.82
    },
    {
      "id": 4,
      "name": "Arroz",
      "score": 0.78
    }
  ],
  "personalized": true
}
```

**Exemplo**:
```bash
curl http://localhost:5001/recommend/user/123
```

---

#### 4. Recomendações por Categoria

```http
GET /recommend/category/<category>
```

**Parâmetros**:
- `category` (string): Nome da categoria

**Categorias Disponíveis**:
- `frutas`
- `legumes`
- `laticinios`
- `graos`

**Descrição**: Retorna recomendações de uma categoria específica.

**Resposta de Sucesso** (200 OK):
```json
{
  "category": "frutas",
  "recommendations": [
    "Banana",
    "Maçã",
    "Laranja",
    "Uva"
  ]
}
```

**Exemplo**:
```bash
curl http://localhost:5001/recommend/category/frutas
```

---

## 🔒 Autenticação

**Status Atual**: Nenhuma autenticação implementada (desenvolvimento).

**Roadmap**:
- [ ] JWT Authentication
- [ ] OAuth2
- [ ] API Keys
- [ ] Rate Limiting

---

## 📊 Códigos de Status HTTP

| Código | Significado | Quando Usar |
|--------|-------------|-------------|
| 200 | OK | Requisição bem-sucedida |
| 201 | Created | Recurso criado com sucesso |
| 400 | Bad Request | Dados inválidos |
| 401 | Unauthorized | Autenticação necessária |
| 404 | Not Found | Recurso não encontrado |
| 500 | Internal Server Error | Erro no servidor |

---

## 🧪 Testando as APIs

### Usando cURL

```bash
# Health checks
curl http://localhost:5000/api/health
curl http://localhost:5001/health

# Recomendações
curl http://localhost:5000/api/recommendations
curl http://localhost:5001/recommend
curl http://localhost:5001/recommend/user/1
curl http://localhost:5001/recommend/category/frutas

# Produtos
curl http://localhost:5000/api/products
curl -X POST http://localhost:5000/api/products \
  -H "Content-Type: application/json" \
  -d '{"name":"Produto Teste","price":9.99}'
```

### Usando Postman

1. Importe a collection (se disponível)
2. Configure a base URL
3. Teste cada endpoint

### Usando o App

O frontend consome todas as APIs automaticamente através de `frontend/src/api/api.ts`.

---

## 🚧 Próximas Features

### Backend
- [ ] Autenticação e autorização
- [ ] CRUD completo de produtos
- [ ] Sistema de favoritos
- [ ] Histórico de compras
- [ ] Filtros e busca avançada

### Recommender
- [ ] Algoritmos de ML reais
- [ ] Collaborative Filtering
- [ ] Content-Based Filtering
- [ ] Treinamento de modelos
- [ ] Cache de recomendações

---

## 📞 Suporte

Dúvidas sobre a API? Consulte:
- README.md principal
- Issues no GitHub
- Documentação inline no código

---

## 🔄 Versionamento

**Versão Atual**: 1.0.0

**Changelog**:
- v1.0.0 (2025-10-05): Release inicial
  - Health checks
  - CRUD básico
  - Recomendações mockadas
  - Integração entre serviços
