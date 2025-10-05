import axios from 'axios';

// Altere este IP para o IP da sua máquina local
// Para descobrir: Windows (ipconfig), Mac/Linux (ifconfig ou ip addr)
const API_URL = 'http://192.168.1.120:5000/api';

export const api = axios.create({
  baseURL: API_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health Check
export const getHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Erro ao verificar health:', error);
    throw error;
  }
};

// Recomendações
export const getRecommendations = async () => {
  try {
    const response = await api.get('/recommendations');
    return response.data;
  } catch (error) {
    console.error('Erro ao buscar recomendações:', error);
    throw error;
  }
};

// Produtos
export const getProducts = async () => {
  try {
    const response = await api.get('/products');
    return response.data;
  } catch (error) {
    console.error('Erro ao buscar produtos:', error);
    throw error;
  }
};

// Supermercados
export const getMarkets = async () => {
  try {
    const response = await api.get('/markets');
    return response.data;
  } catch (error) {
    console.error('Erro ao buscar mercados:', error);
    throw error;
  }
};

// Promoções
export const getPromotions = async () => {
  try {
    const response = await api.get('/promotions');
    return response.data;
  } catch (error) {
    console.error('Erro ao buscar promoções:', error);
    throw error;
  }
};

export default api;
