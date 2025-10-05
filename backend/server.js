import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import dotenv from "dotenv";
import axios from "axios";

dotenv.config();

const app = express();

// Middlewares
app.use(express.json());
app.use(cors());

// MongoDB Connection
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log("✅ MongoDB conectado com sucesso!"))
  .catch((err) => console.error("❌ Erro ao conectar MongoDB:", err));

// Health Check Route
app.get("/api/health", (req, res) => {
  res.json({ 
    status: "ok", 
    message: "Backend Aproveita está rodando!",
    timestamp: new Date().toISOString()
  });
});

// Route to get recommendations from Flask ML API
app.get("/api/recommendations", async (req, res) => {
  try {
    const response = await axios.get(`${process.env.RECOMMENDER_API}/recommend`);
    res.json(response.data);
  } catch (error) {
    console.error("Erro ao buscar recomendações:", error.message);
    res.status(500).json({ 
      error: "Erro ao buscar recomendações",
      message: error.message 
    });
  }
});

// Products Routes
const productsRouter = express.Router();

productsRouter.get("/", (req, res) => {
  res.json({ 
    message: "Lista de produtos",
    products: []
  });
});

productsRouter.post("/", (req, res) => {
  res.json({ 
    message: "Produto criado com sucesso",
    product: req.body
  });
});

app.use("/api/products", productsRouter);

// Markets Routes
const marketsRouter = express.Router();

marketsRouter.get("/", (req, res) => {
  res.json({ 
    message: "Lista de supermercados",
    markets: []
  });
});

app.use("/api/markets", marketsRouter);

// Promotions Routes
const promotionsRouter = express.Router();

promotionsRouter.get("/", (req, res) => {
  res.json({ 
    message: "Lista de promoções",
    promotions: []
  });
});

app.use("/api/promotions", promotionsRouter);

// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Servidor backend rodando na porta ${PORT}`);
  console.log(`📍 Health check: http://localhost:${PORT}/api/health`);
});
