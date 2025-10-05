# Frontend - Aproveita App

## Descrição

Aplicativo móvel em React Native para consumidores descobrirem e aproveitarem promoções.

## Tecnologias

- React Native 0.73
- TypeScript
- React Navigation
- Axios

## Instalação

```bash
npm install
```

## Configuração

Edite `src/api/api.ts` e configure o IP do seu backend:

```typescript
const API_URL = 'http://SEU_IP_LOCAL:5000/api';
```

**Como descobrir o IP:**
- Windows: `ipconfig`
- Mac/Linux: `ifconfig` ou `ip addr`

## Executar

### Android

```bash
# Iniciar Metro Bundler
npm start

# Em outro terminal
npm run android
```

### iOS (Mac apenas)

```bash
# Instalar pods
cd ios && pod install && cd ..

# Executar
npm run ios
```

## Estrutura

```
frontend/
├── android/           # Configuração Android
├── ios/              # Configuração iOS
├── src/
│   ├── api/         # Cliente HTTP
│   ├── screens/     # Telas
│   │   ├── HomeScreen.tsx
│   │   ├── RecommendationsScreen.tsx
│   │   ├── MarketsScreen.tsx
│   │   └── ProfileScreen.tsx
│   └── components/  # Componentes reutilizáveis
├── App.tsx          # Entry point
└── package.json
```

## Telas

### 🏠 Home
- Promoções destacadas
- Categorias
- Status do backend

### ✨ Recomendações
- Produtos recomendados pelo ML
- Pull to refresh
- Match score

### 🛒 Mercados
- Lista de supermercados próximos
- Distância
- Quantidade de promoções

### 👤 Perfil
- Configurações do usuário
- Preferências
- Favoritos

## Troubleshooting

### Build falhou

```bash
# Limpar cache
npx react-native clean
cd android && ./gradlew clean && cd ..

# Rebuild
npm run android
```

### Erro de conexão com backend

1. Verifique se o backend está rodando
2. Confirme o IP em `src/api/api.ts`
3. Teste: `curl http://SEU_IP:5000/api/health`

### Metro Bundler não inicia

```bash
# Limpar cache do Metro
npx react-native start --reset-cache
```

## Próximos passos

- [ ] Implementar autenticação
- [ ] Adicionar favorites/watchlist
- [ ] Push notifications
- [ ] Modo offline
- [ ] Testes unitários e E2E
- [ ] Dark mode
