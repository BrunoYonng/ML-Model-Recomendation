import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import axios from "axios";
dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/aproveita";
const PORT = process.env.PORT || 5000;
const RECOMMENDER_API = process.env.RECOMMENDER_API || "http://127.0.0.1:5001";

async function connectToMongo() {
  try {
    await mongoose.connect(MONGO_URI);
    console.log("✅ MongoDB conectado!");
  } catch (err) {
    console.error("❌ Erro MongoDB:", err?.message || err);
  }
}

connectToMongo();

app.get("/api/health", (req, res) => res.json({ status: "ok" }));

app.get("/api/recommendations", async (req, res) => {
  try {
    const response = await axios.get(`${RECOMMENDER_API}/recommend`);
    return res.json(response.data);
  } catch (error) {
    const message = error?.response?.data || error?.message || "Erro ao obter recomendações";
    return res.status(502).json({ error: message });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Servidor a correr na porta ${PORT}`);
});
