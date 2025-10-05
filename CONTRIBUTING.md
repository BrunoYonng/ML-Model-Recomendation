# 🤝 Guia de Contribuição - Aproveita

Obrigado por considerar contribuir para o Aproveita! Este documento fornece diretrizes para contribuir com o projeto.

## 📋 Índice

- [Como Contribuir](#como-contribuir)
- [Configuração do Ambiente](#configuração-do-ambiente)
- [Processo de Desenvolvimento](#processo-de-desenvolvimento)
- [Padrões de Código](#padrões-de-código)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

---

## Como Contribuir

Existem várias formas de contribuir:

### 🐛 Reportar Bugs

Se encontrou um bug, abra uma issue incluindo:
- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs atual
- Screenshots (se aplicável)
- Ambiente (SO, versões, etc.)

### ✨ Sugerir Features

Para sugerir novas funcionalidades:
- Verifique se já não existe uma issue similar
- Descreva claramente a feature
- Explique por que seria útil
- Forneça exemplos de uso

### 📝 Melhorar Documentação

Documentação sempre pode melhorar:
- Corrigir typos
- Adicionar exemplos
- Esclarecer instruções confusas
- Traduzir documentação

### 💻 Contribuir com Código

1. Fork o projeto
2. Crie uma branch para sua feature
3. Faça suas alterações
4. Teste tudo
5. Submeta um Pull Request

---

## Configuração do Ambiente

Siga o [INSTALL.md](INSTALL.md) para configurar o ambiente de desenvolvimento.

---

## Processo de Desenvolvimento

### 1. Fork e Clone

```bash
# Fork no GitHub primeiro, depois:
git clone https://github.com/SEU_USUARIO/aproveita.git
cd aproveita
```

### 2. Criar Branch

```bash
git checkout -b feature/nome-da-feature
# ou
git checkout -b fix/nome-do-bug
```

### 3. Fazer Alterações

- Siga os padrões de código
- Adicione testes se aplicável
- Atualize documentação se necessário

### 4. Testar

```bash
# Backend
cd backend
npm test

# Frontend
cd frontend
npm test

# Recommender
cd recommender
python -m pytest
```

### 5. Commit

```bash
git add .
git commit -m "feat: adiciona nova funcionalidade"
```

### 6. Push

```bash
git push origin feature/nome-da-feature
```

### 7. Pull Request

Abra um PR no GitHub com:
- Descrição clara das mudanças
- Referência a issues relacionadas
- Screenshots (se UI)
- Checklist de testes

---

## Padrões de Código

### JavaScript/TypeScript

```javascript
// Use const/let, nunca var
const apiUrl = 'http://localhost:5000';
let count = 0;

// Arrow functions
const fetchData = async () => {
  // code
};

// Destructuring
const {data, error} = await api.get('/endpoint');

// Template literals
console.log(`Count: ${count}`);
```

### Python

```python
# PEP 8
# Snake case para funções e variáveis
def get_recommendations():
    user_id = 123
    return results

# Classes em PascalCase
class RecommenderEngine:
    pass

# Type hints
def calculate_score(items: list) -> float:
    return 0.0
```

### React Native

```typescript
// Functional components
const HomeScreen: React.FC = () => {
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    loadData();
  }, []);
  
  return (
    <View style={styles.container}>
      <Text>Home</Text>
    </View>
  );
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
});
```

---

## Commit Guidelines

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

### Tipos

- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `docs`: Documentação
- `style`: Formatação (não afeta código)
- `refactor`: Refatoração
- `test`: Testes
- `chore`: Tarefas de build/config

### Exemplos

```bash
feat: adiciona tela de favoritos
fix: corrige crash ao carregar promoções
docs: atualiza README com novas instruções
style: formata código com prettier
refactor: reorganiza estrutura de pastas
test: adiciona testes para API de produtos
chore: atualiza dependências do projeto
```

### Formato

```
tipo(escopo): descrição curta

Descrição mais detalhada (opcional)

Fixes #123
```

---

## Pull Request Process

### Checklist

Antes de submeter um PR, verifique:

- [ ] Código segue os padrões do projeto
- [ ] Todos os testes passam
- [ ] Adicionou testes para novas funcionalidades
- [ ] Documentação foi atualizada
- [ ] Commits seguem o padrão Conventional Commits
- [ ] Branch está atualizada com main
- [ ] Não há conflitos
- [ ] PR tem descrição clara

### Template de PR

```markdown
## Descrição
Breve descrição das mudanças.

## Tipo de Mudança
- [ ] Bug fix
- [ ] Nova feature
- [ ] Breaking change
- [ ] Documentação

## Como Testar
1. Passo 1
2. Passo 2
3. Verificar resultado esperado

## Screenshots
(se aplicável)

## Checklist
- [ ] Código revisado
- [ ] Testes passando
- [ ] Documentação atualizada

## Issues Relacionadas
Closes #123
```

---

## Estrutura de Branches

- `main`: Produção (protegida)
- `develop`: Desenvolvimento
- `feature/*`: Novas funcionalidades
- `fix/*`: Correções
- `hotfix/*`: Correções urgentes
- `release/*`: Preparação de release

---

## Code Review

Pull requests passarão por code review. Espere:

- Feedback construtivo
- Sugestões de melhorias
- Solicitação de mudanças (se necessário)

Seja paciente e receptivo ao feedback!

---

## Testes

### Backend (Jest)

```javascript
describe('API Health Check', () => {
  it('should return status ok', async () => {
    const response = await request(app).get('/api/health');
    expect(response.status).toBe(200);
    expect(response.body.status).toBe('ok');
  });
});
```

### Frontend (Jest + React Native Testing Library)

```typescript
import {render, fireEvent} from '@testing-library/react-native';

describe('HomeScreen', () => {
  it('renders correctly', () => {
    const {getByText} = render(<HomeScreen />);
    expect(getByText('Aproveita')).toBeTruthy();
  });
});
```

### Recommender (Pytest)

```python
def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'running'
```

---

## Comunidade

- Seja respeitoso e inclusivo
- Ajude outros desenvolvedores
- Compartilhe conhecimento
- Celebre conquistas

---

## Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a mesma licença do projeto (MIT).

---

## Dúvidas?

- Abra uma issue
- Entre em contato com os maintainers
- Consulte a documentação

**Obrigado por contribuir! 🎉**
