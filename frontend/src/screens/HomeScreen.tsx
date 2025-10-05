import React, {useEffect, useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import {getHealth, getPromotions} from '../api/api';

const HomeScreen = () => {
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<string>('Verificando...');
  const [promotions, setPromotions] = useState<any[]>([]);

  useEffect(() => {
    checkApiHealth();
    loadPromotions();
  }, []);

  const checkApiHealth = async () => {
    try {
      const health = await getHealth();
      setApiStatus(health.status === 'ok' ? '✅ Online' : '❌ Offline');
    } catch (error) {
      setApiStatus('❌ Offline');
      console.error('API não está respondendo:', error);
    }
  };

  const loadPromotions = async () => {
    setLoading(true);
    try {
      const data = await getPromotions();
      setPromotions(data.promotions || []);
    } catch (error) {
      console.error('Erro ao carregar promoções:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>🛒 Aproveita</Text>
        <Text style={styles.subtitle}>Promoções em tempo real</Text>
        <View style={styles.statusBadge}>
          <Text style={styles.statusText}>Backend: {apiStatus}</Text>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>🔥 Promoções Destacadas</Text>
        {loading ? (
          <ActivityIndicator size="large" color="#e91e63" />
        ) : promotions.length > 0 ? (
          promotions.map((promo, index) => (
            <TouchableOpacity key={index} style={styles.card}>
              <Text style={styles.cardTitle}>{promo.name}</Text>
              <Text style={styles.cardDiscount}>{promo.discount}% OFF</Text>
            </TouchableOpacity>
          ))
        ) : (
          <View style={styles.emptyState}>
            <Text style={styles.emptyText}>
              Nenhuma promoção disponível no momento
            </Text>
            <Text style={styles.emptySubtext}>
              Estamos buscando as melhores ofertas para você! 🎯
            </Text>
          </View>
        )}
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>📱 Categorias</Text>
        <View style={styles.categoriesGrid}>
          {['Frutas', 'Legumes', 'Laticínios', 'Grãos', 'Bebidas', 'Carnes'].map(
            (category, index) => (
              <TouchableOpacity key={index} style={styles.categoryCard}>
                <Text style={styles.categoryText}>{category}</Text>
              </TouchableOpacity>
            ),
          )}
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#e91e63',
    padding: 24,
    paddingTop: 48,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 16,
    color: '#fff',
    opacity: 0.9,
  },
  statusBadge: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    alignSelf: 'flex-start',
    marginTop: 12,
  },
  statusText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  section: {
    padding: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
  },
  card: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  cardDiscount: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#e91e63',
  },
  emptyState: {
    backgroundColor: '#fff',
    padding: 32,
    borderRadius: 12,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#666',
    textAlign: 'center',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
  },
  categoriesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  categoryCard: {
    backgroundColor: '#fff',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 1},
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  categoryText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
});

export default HomeScreen;
