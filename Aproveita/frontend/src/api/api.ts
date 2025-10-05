import axios from 'axios';

const API_URL = 'http://192.168.1.120:5000/api';

export const getHealth = async () => {
  const res = await axios.get(`${API_URL}/health`);
  return res.data;
};
