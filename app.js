import express, { json } from "express";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import cors from "cors";
import imageRoutes from "./routes/imageRoutes.js";
import { fileURLToPath } from "url";
import morgan from "morgan";
import { dirname, join } from "path";
import { getRotationStats } from "./services/geminiService.js";
import { DetectionService } from "./services/DetectionService.js";
import { createServer } from "http";
import { initializeSocketServer } from "./socket/SocketServer.js";

// Importar dotenv para manejar variables de entorno
import dotenv from "dotenv";
dotenv.config();

const app = express();

app.set("trust proxy", 1); // Confía solo en el primer proxy

// Seguridad
app.use(helmet());
app.use(cors());
app.use(json());
// Registro de solicitudes
app.use(morgan("dev"));

// Limitar solicitudes por IP
const limiter = rateLimit({
  windowMs: 1 * 60 * 1000,
  max: 120, // Aumentado para permitir análisis cada 500ms (hasta 120 por minuto)
});
app.use(limiter);

// Rutas
app.use("/api/image", imageRoutes);

// Endpoint para ver estadísticas de rotación
app.get("/api/stats", (req, res) => {
  const stats = getRotationStats();
  res.json({
    message: "Estadísticas de rotación de API keys",
    stats,
    timestamp: new Date().toISOString()
  });
});

// Convertir `import.meta.url` a __dirname (compatible con ES Modules)
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
// El frontend se sirve desde front-irix, no desde el backend

const PORT = 3000;

// Crear servidor HTTP para Socket.IO
const server = createServer(app);

// Inicializar Socket.IO para detección de placas en tiempo real
await initializeSocketServer(server);

server.listen(PORT, async () => {
  console.log(`🚀 Servidor HTTP corriendo en http://localhost:${PORT}`);
  console.log(`🔌 Socket.IO activo en http://localhost:${PORT}`);
  
  // Inicializar servicio de detección de placas
  console.log('⚡ Inicializando modelo de detección de placas...');
  const detectionService = new DetectionService();
  await detectionService.initialize();
  console.log('✅ Sistema de detección de placas listo!');
});
