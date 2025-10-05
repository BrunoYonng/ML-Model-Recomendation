# 🚀 Guia de Instalação Rápida - Aproveita

## Pré-requisitos

Instale tudo nesta ordem:

### 1. Node.js e npm
- Download: https://nodejs.org/ (versão LTS 20+)
- Verificar: `node -v` e `npm -v`

### 2. Python
- Download: https://www.python.org/downloads/ (3.10+)
- **IMPORTANTE**: Marque "Add Python to PATH" durante instalação
- Verificar: `python --version`

### 3. MongoDB
- Download: https://www.mongodb.com/try/download/community
- Instalar como serviço (opção durante instalação)
- Verificar: `mongod --version`

### 4. JDK 17
- Download: https://www.oracle.com/java/technologies/downloads/#java17
- Configurar JAVA_HOME
- Verificar: `java -version`

### 5. Android Studio
- Download: https://developer.android.com/studio
- Durante instalação, incluir:
  - Android SDK
  - Android SDK Platform
  - Android Virtual Device

#### Configurar Android SDK

1. Abra Android Studio
2. More Actions → SDK Manager
3. SDK Platforms: Marque Android 14.0 (API 34)
4. SDK Tools: Marque:
   - Android SDK Build-Tools
   - Android SDK Command-line Tools
   - Android Emulator
   - Android SDK Platform-Tools

#### Configurar Variáveis de Ambiente

**Windows:**
```
ANDROID_HOME = C:\Users\SeuUsuario\AppData\Local\Android\Sdk
PATH += %ANDROID_HOME%\platform-tools
PATH += %ANDROID_HOME%\emulator
PATH += %ANDROID_HOME%\tools
PATH += %ANDROID_HOME%\tools\bin
```

**Mac/Linux:**
```bash
# Adicionar ao ~/.bashrc ou ~/.zshrc
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/emulator
export PATH=$PATH:$ANDROID_HOME/platform-tools
```

#### Criar Emulador

1. Android Studio → More Actions → Virtual Device Manager
2. Create Device
3. Selecione: Pixel 5 ou similar
4. System Image: Android 14.0 (API 34)
5. Finish

## Instalação do Projeto

### Passo 1: Clonar/Baixar o Projeto

```bash
git clone <repository-url>
cd Aproveita
```

### Passo 2: Instalar Backend

```bash
cd backend
npm install
cd ..
```

### Passo 3: Instalar Recommender

```bash
cd recommender

# Criar ambiente virtual
python -m venv venv

# Ativar (IMPORTANTE!)
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Desativar por enquanto
deactivate

cd ..
```

### Passo 4: Instalar Frontend

```bash
cd frontend
npm install
cd ..
```

## Configuração

### 1. Descobrir seu IP local

**Windows:**
```bash
ipconfig
# Procure por "IPv4 Address" na seção do seu adaptador de rede
# Ex: 192.168.1.100
```

**Mac:**
```bash
ifconfig | grep "inet "
# Procure por algo como: inet 192.168.1.100
```

**Linux:**
```bash
ip addr show
# Procure por: inet 192.168.1.100/24
```

### 2. Configurar o IP no Frontend

Edite o arquivo: `frontend/src/api/api.ts`

```typescript
const API_URL = 'http://192.168.1.XXX:5000/api'; // Substitua XXX pelo seu IP
```

### 3. Verificar MongoDB

```bash
# Windows
net start MongoDB

# Mac
brew services start mongodb-community

# Linux
sudo systemctl start mongod
```

## Executar o Projeto

### Terminal 1 - Backend

```bash
cd backend
npm run dev
```

Aguarde ver: `✅ MongoDB conectado!` e `🚀 Servidor backend rodando na porta 5000`

### Terminal 2 - Recommender

```bash
cd recommender

# ATIVAR AMBIENTE VIRTUAL PRIMEIRO!
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

python app.py
```

Aguarde ver: `🤖 Iniciando serviço de recomendações...`

### Terminal 3 - Android (Emulador)

```bash
# Iniciar emulador (ou conectar dispositivo físico)
# No Android Studio: AVD Manager → Play (▶️)

# Verificar que emulador está rodando
adb devices
# Deve mostrar um dispositivo
```

### Terminal 4 - Frontend

```bash
cd frontend
npm run android
```

Aguarde o build e o app abrirá no emulador!

## Verificação

1. Backend: http://localhost:5000/api/health
2. Recommender: http://localhost:5001/health
3. App no emulador deve mostrar a tela inicial

## Problemas Comuns

### "Could not connect to development server"

**Causa**: IP incorreto ou backend não está rodando.

**Solução**:
1. Verifique se backend está rodando (Terminal 1)
2. Confirme o IP em `frontend/src/api/api.ts`
3. No emulador, teste: `adb shell ping SEU_IP`

### "SDK location not found"

**Solução**: Crie `frontend/android/local.properties`:
```
sdk.dir=C:\\Users\\SeuUsuario\\AppData\\Local\\Android\\Sdk
# ou no Mac/Linux:
# sdk.dir=/Users/SeuUsuario/Library/Android/sdk
```

### "Python não é reconhecido"

**Solução**: Reinstale Python e marque "Add Python to PATH"

### Build Android falhou

```bash
cd frontend
npx react-native doctor
# Siga as instruções para corrigir problemas
```

## Checklist Final

- [ ] Node.js instalado
- [ ] Python instalado
- [ ] MongoDB rodando
- [ ] JDK 17 instalado
- [ ] Android Studio configurado
- [ ] Emulador criado e funcionando
- [ ] Backend instalado e rodando
- [ ] Recommender instalado e rodando
- [ ] Frontend instalado
- [ ] IP configurado no frontend
- [ ] App abriu no emulador

## 🎉 Pronto!

Se todos os checkboxes estão marcados, o projeto está funcionando!

## Suporte

Se encontrar problemas:
1. Consulte o README.md principal
2. Verifique a seção Troubleshooting
3. Abra uma issue no GitHub
