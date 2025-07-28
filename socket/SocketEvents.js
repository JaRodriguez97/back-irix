/**
 * Definición de eventos Socket.IO para detección de placas en tiempo real
 * Centraliza todos los eventos y constantes del sistema
 */

// Eventos del cliente hacia el servidor
const CLIENT_EVENTS = {
  CONNECT: "connect",
  DISCONNECT: "disconnect",
  ANALYZE_IMAGE: "analyze-image",
  JOIN_ROOM: "join-room",
  LEAVE_ROOM: "leave-room",
};

// Eventos del servidor hacia el cliente
const SERVER_EVENTS = {
  ANALYSIS_RESULT: "analysis-result",
  ANALYSIS_ERROR: "analysis-error",
  SERVER_STATUS: "server-status",
  CONNECTION_CONFIRMED: "connection-confirmed",
};

// Configuración de Socket.IO
const SOCKET_CONFIG = {
  // Puerto del servidor Socket.IO (si es diferente al HTTP)
  PORT: process.env.SOCKET_PORT || 3001,

  // Configuración CORS para desarrollo
  CORS: "*" /* {
        origin: process.env.FRONTEND_URL || "http://localhost:4200",
        methods: ["GET", "POST"],
        credentials: true
    } */,

  // Configuración de conexión
  CONNECTION: {
    pingTimeout: 60000,
    pingInterval: 25000,
    maxHttpBufferSize: 1e7, // 10MB para imágenes
    transports: ["polling", "websocket"], // Socket.IO maneja automáticamente
  },
};

// Configuración de análisis de imágenes
const IMAGE_CONFIG = {
  MAX_SIZE: 5 * 1024 * 1024, // 5MB máximo por imagen
  ALLOWED_FORMATS: ["image/jpeg", "image/png", "image/webp"],
  PREFERRED_FORMAT: "image/webp", // Formato preferido para mejor performance
  WEBP_QUALITY: 0.9, // Calidad WebP (0.9 = 90%)
  JPEG_FALLBACK_QUALITY: 0.85, // Calidad JPEG como fallback
  PROCESSING_TIMEOUT: 10000, // 10 segundos timeout
  EXPECTED_DIMENSIONS: {
    width: 300,
    height: 300,
  },
  // Configuraciones de optimización
  OPTIMIZATION: {
    ENABLE_WEBP: true,
    ENABLE_COMPRESSION: true,
    ENABLE_MEMORY_CLEANUP: true,
    TARGET_FILE_SIZE: 15 * 1024 // 15KB target (vs 36KB actual)
  }
};

// Estados del servidor
const SERVER_STATUS = {
  READY: "ready",
  PROCESSING: "processing",
  ERROR: "error",
  MAINTENANCE: "maintenance",
};

// Códigos de error estandarizados
const ERROR_CODES = {
  INVALID_IMAGE: "INVALID_IMAGE",
  IMAGE_TOO_LARGE: "IMAGE_TOO_LARGE",
  PROCESSING_TIMEOUT: "PROCESSING_TIMEOUT",
  MODEL_NOT_LOADED: "MODEL_NOT_LOADED",
  INTERNAL_ERROR: "INTERNAL_ERROR",
  INVALID_FORMAT: "INVALID_FORMAT",
};

export {
  CLIENT_EVENTS,
  SERVER_EVENTS,
  SOCKET_CONFIG,
  IMAGE_CONFIG,
  SERVER_STATUS,
  ERROR_CODES,
};
