import React, {useEffect, useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import {getMarkets} from '../api/api';

const MarketsScreen = () => {
  const [loading, setLoading] = useState(false);
  const [markets, setMarkets] = useState<any[]>([]);

  useEffect(() => {
    loadMarkets();
  }, []);

  const loadMarkets = async () => {
    setLoading(true);
    try {
      const data = await getMarkets();
      setMarkets(data.markets || []);
    } catch (error) {
      console.error('Erro ao carregar mercados:', error);
    } finally {
      setLoading(false);
    }
  };

  // Mock data para demonstração
  const mockMarkets = [
    {id: 1, name: 'Continente', distance: '1.2 km', promotions: 45},
    {id: 2, name: 'Pingo Doce', distance: '2.5 km', promotions: 38},
    {id: 3, name: 'Lidl', distance: '3.1 km', promotions: 52},
    {id: 4, name: 'Auchan', distance: '4.8 km', promotions: 29},
    {id: 5, name: 'Minipreço', distance: '0.8 km', promotions: 21},
  ];

  const displayMarkets = markets.length > 0 ? markets : mockMarkets;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>🛒 Supermercados</Text>
        <Text style={styles.subtitle}>Próximos de você</Text>
      </View>

      <ScrollView style={styles.content}>
        {loading ? (
          <ActivityIndicator size="large" color="#e91e63" />
        ) : (
          <>
            {displayMarkets.map(market => (
              <TouchableOpacity key={market.id} style={styles.card}>
                <View style={styles.cardLeft}>
                  <Text style={styles.marketIcon}>🏪</Text>
                </View>
                <View style={styles.cardCenter}>
                  <Text style={styles.marketName}>{market.name}</Text>
                  <Text style={styles.marketDistance}>📍 {market.distance}</Text>
                </View>
                <View style={styles.cardRight}>
                  <View style={styles.promotionsBadge}>
                    <Text style={styles.promotionsCount}>
                      {market.promotions}
                    </Text>
                    <Text style={styles.promotionsLabel}>promoções</Text>
                  </View>
                </View>
              </TouchableOpacity>
            ))}
          </>
        )}
      </ScrollView>
    </View>
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
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.9,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardLeft: {
    marginRight: 16,
  },
  marketIcon: {
    fontSize: 32,
  },
  cardCenter: {
    flex: 1,
  },
  marketName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  marketDistance: {
    fontSize: 14,
    color: '#666',
  },
  cardRight: {
    alignItems: 'center',
  },
  promotionsBadge: {
    backgroundColor: '#e91e63',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 12,
    minWidth: 60,
    alignItems: 'center',
  },
  promotionsCount: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  promotionsLabel: {
    fontSize: 10,
    color: '#fff',
    opacity: 0.9,
  },
});

export default MarketsScreen;
