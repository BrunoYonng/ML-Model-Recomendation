# 🛒 Aproveita - Sistema Inteligente de Recomendação de Promoções

<div align="center">

![Aproveita Logo](https://img.shields.io/badge/Aproveita-v1.0.0-e91e63?style=for-the-badge)
![React Native](https://img.shields.io/badge/React_Native-0.73-61DAFB?style=for-the-badge&logo=react)
![Node.js](https://img.shields.io/badge/Node.js-20+-339933?style=for-the-badge&logo=node.js)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask)
![MongoDB](https://img.shields.io/badge/MongoDB-Latest-47A248?style=for-the-badge&logo=mongodb)

**Um app inspirado na Netflix para descobrir promoções de supermercados em tempo real!**

</div>

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Tecnologias](#-tecnologias)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Configuração](#-configuração)
- [Executar o Projeto](#-executar-o-projeto)
- [APIs Disponíveis](#-apis-disponíveis)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Sobre o Projeto

O **Aproveita** é uma plataforma que conecta consumidores a promoções de supermercados de forma inteligente e personalizada. Com machine learning, o sistema recomenda produtos com base nas preferências do usuário, proporcionando uma experiência similar à Netflix, mas para compras de supermercado.

### ✨ Funcionalidades

- 📱 **App Mobile** com interface moderna e intuitiva
- 🤖 **Recomendações inteligentes** com ML
- 🏪 **Múltiplos supermercados** integrados
- 🔔 **Alertas em tempo real** de promoções
- 📍 **Localização** dos mercados mais próximos
- ❤️ **Favoritos** e listas personalizadas

---

## 🚀 Tecnologias

### Frontend
- **React Native CLI** (TypeScript)
- **React Navigation** (Stack & Bottom Tabs)
- **Axios** para chamadas HTTP

### Backend
- **Node.js** com Express
- **MongoDB** para persistência
- **Mongoose** ODM

### Recommender (ML)
- **Flask** (Python)
- **Pandas** para manipulação de dados
- **Scikit-learn** para algoritmos de ML

---

## 📂 Estrutura do Projeto

```
Aproveita/
├── frontend/                 # React Native App
│   ├── android/             # Configuração Android
│   ├── src/
│   │   ├── api/            # Cliente API
│   │   ├── screens/        # Telas do app
│   │   └── components/     # Componentes reutilizáveis
│   ├── App.tsx             # Entry point
│   └── package.json
│
├── backend/                 # Node.js API
│   ├── server.js           # Servidor Express
│   ├── .env                # Variáveis de ambiente
│   └── package.json
│
├── recommender/             # Flask ML API
│   ├── app.py              # Serviço de recomendações
│   ├── requirements.txt    # Dependências Python
│   └── venv/               # Ambiente virtual
│
├── .env                     # Configuração global
└── README.md
```

---

## ✅ Pré-requisitos

Antes de começar, certifique-se de ter instalado:

### Obrigatórios
- **Node.js** >= 20.0.0 ([Download](https://nodejs.org/))
- **npm** >= 10.0.0
- **Python** >= 3.10 ([Download](https://www.python.org/))
- **MongoDB** ([Download](https://www.mongodb.com/try/download/community))

### Para Android
- **JDK** 17 ([Download](https://www.oracle.com/java/technologies/downloads/))
- **Android Studio** ([Download](https://developer.android.com/studio))
- **Android SDK** (instalado via Android Studio)
- Emulador Android ou dispositivo físico

### Verificar instalações

```bash
node -v        # >= v20.0.0
npm -v         # >= 10.0.0
python --version  # >= 3.10
java -version  # 17
```

---

## 📦 Instalação

### 1️⃣ Clonar o Repositório

```bash
git clone <repository-url>
cd Aproveita
```

### 2️⃣ Backend (Node.js)

```bash
cd backend
npm install
```

### 3️⃣ Recommender (Flask)

```bash
cd ../recommender

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 4️⃣ Frontend (React Native)

```bash
cd ../frontend
npm install

# Limpar cache (se necessário)
npx react-native clean
```

---

## ⚙️ Configuração

### 1. MongoDB

Certifique-se de que o MongoDB está rodando:

```bash
# Windows (se instalado como serviço)
net start MongoDB

# Mac
brew services start mongodb-community

# Linux
sudo systemctl start mongod
```

### 2. Configurar .env

Edite o arquivo `/workspace/.env` e ajuste as configurações:

```env
# Backend
BACKEND_PORT=5000
MONGO_URI=mongodb://localhost:27017/aproveita

# Recommender
RECOMMENDER_PORT=5001

# Frontend - IMPORTANTE: Coloque o IP da sua máquina
API_URL=http://SEU_IP_LOCAL:5000/api
```

**Como descobrir seu IP local:**

```bash
# Windows
ipconfig

# Mac/Linux
ifconfig
# ou
ip addr show
```

Procure por algo como `192.168.1.XXX` e substitua em `/workspace/frontend/src/api/api.ts`:

```typescript
const API_URL = 'http://192.168.1.XXX:5000/api';
```

### 3. Configurar Android

Abra o Android Studio e:
1. Abra o AVD Manager
2. Crie um dispositivo virtual (ex: Pixel 5, API 34)
3. Inicie o emulador

Ou conecte um dispositivo físico via USB e ative o modo desenvolvedor.

---

## 🎬 Executar o Projeto

### Opção 1: Terminal Separados (Recomendado)

**Terminal 1 - Backend:**
```bash
cd backend
npm run dev
# Aguarde: "🚀 Servidor backend rodando na porta 5000"
```

**Terminal 2 - Recommender:**
```bash
cd recommender
# Ativar ambiente virtual primeiro!
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

python app.py
# Aguarde: "🤖 Iniciando serviço de recomendações..."
```

**Terminal 3 - Frontend:**
```bash
cd frontend

# Iniciar Metro Bundler
npm start

# Em outro terminal, executar no Android
npm run android
```

### Opção 2: Scripts de Conveniência

Você pode criar scripts no `package.json` da raiz (futuro):

```json
{
  "scripts": {
    "backend": "cd backend && npm run dev",
    "recommender": "cd recommender && python app.py",
    "frontend": "cd frontend && npm run android"
  }
}
```

---

## 🔌 APIs Disponíveis

### Backend (http://localhost:5000)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/api/health` | Health check |
| GET | `/api/recommendations` | Recomendações via ML |
| GET | `/api/products` | Lista de produtos |
| GET | `/api/markets` | Lista de supermercados |
| GET | `/api/promotions` | Promoções ativas |

### Recommender (http://localhost:5001)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Health check |
| GET | `/recommend` | Recomendações gerais |
| GET | `/recommend/user/<id>` | Recomendações personalizadas |
| GET | `/recommend/category/<cat>` | Por categoria |

---

## 🐛 Troubleshooting

### ❌ "Could not connect to development server"

**Solução:** Certifique-se de que o IP em `frontend/src/api/api.ts` está correto.

```bash
# Testar conexão
curl http://SEU_IP:5000/api/health
```

### ❌ Erro no build Android

```bash
cd frontend/android
./gradlew clean
cd ..
npx react-native clean
npm run android
```

### ❌ MongoDB não conecta

Verifique se o serviço está rodando:

```bash
# Ver status
# Windows: services.msc (procure MongoDB)
# Mac: brew services list
# Linux: sudo systemctl status mongod

# Reiniciar
# Windows: net stop MongoDB && net start MongoDB
# Mac: brew services restart mongodb-community
# Linux: sudo systemctl restart mongod
```

### ❌ Porta já em uso

```bash
# Backend (5000)
# Windows: netstat -ano | findstr :5000
# Mac/Linux: lsof -i :5000

# Matar processo
# Windows: taskkill /PID <PID> /F
# Mac/Linux: kill -9 <PID>
```

### ❌ Python venv não ativa

```bash
# Recriar ambiente virtual
cd recommender
rm -rf venv  # ou rmdir /s venv no Windows
python -m venv venv

# Ativar novamente
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

---

## 📱 Capturas de Tela

_(Adicione aqui screenshots do app quando estiver rodando)_

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## 👨‍💻 Autores

Desenvolvido com ❤️ para revolucionar a forma como você economiza!

---

## 📞 Suporte

Encontrou um problema? [Abra uma issue](https://github.com/seu-usuario/aproveita/issues)

---

<div align="center">

### 🌟 Se gostou do projeto, deixe uma estrela!

**Aproveita** - Economize de forma inteligente! 🛒✨

</div>
