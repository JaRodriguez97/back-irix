/**
 * Configuración del Modelo de Detección de Placas Vehiculares
 * TensorFlow.js + Node.js 22
 */

export const MODEL_CONFIG = {
  // Configuración de entrada
  INPUT: {
    WIDTH: 224,           // Entrada estándar para MobileNet
    HEIGHT: 224,
    CHANNELS: 3,          // RGB
    ORIGINAL_SIZE: 300    // Tamaño original de entrada del usuario
  },

  // Configuración del modelo base
  MODEL: {
    NAME: 'mobilenet',
    VERSION: '2.0',
    ALPHA: 1.0,           // Factor de ancho del modelo
    INPUT_RANGE: [0, 1],  // Rango de normalización
    MEAN: [0.485, 0.456, 0.406], // Normalización ImageNet
    STD: [0.229, 0.224, 0.225]
  },

  // Transfer Learning
  TRANSFER_LEARNING: {
    FREEZE_BASE: true,    // Congelar capas base de MobileNet
    CLASSIFIER_UNITS: 1,  // Una sola salida (probabilidad binaria)
    ACTIVATION: 'sigmoid', // Para clasificación binaria
    DROPOUT_RATE: 0.2,
    LEARNING_RATE: 0.001, // Learning rate para transfer learning
    EPOCHS: 10,           // Número de epochs de entrenamiento
    BATCH_SIZE: 32,       // Tamaño de batch para entrenamiento
    VALIDATION_SPLIT: 0.2 // Porcentaje de datos para validación
  },

  // Umbrales y confianza
  PREDICTION: {
    CONFIDENCE_THRESHOLD: 0.5,  // Umbral para considerar placa detectada
    MIN_CONFIDENCE: 0.1,        // Confianza mínima para respuesta válida
    MAX_CONFIDENCE: 0.9         // Confianza máxima esperada
  },

  // Performance y optimización
  PERFORMANCE: {
    TARGET_FPS: 2,
    MAX_INFERENCE_TIME: 200,    // 🚀 OPTIMIZADO: Reducido de 500ms a 200ms
    MAX_MEMORY_MB: 150,         // 🚀 OPTIMIZADO: Reducido de 200MB a 150MB
    BATCH_SIZE: 1,              // Procesamiento individual
    USE_GPU: false,             // Inicialmente CPU only
    // 🚀 NUEVAS OPTIMIZACIONES:
    ENABLE_TENSOR_POOLING: true,
    ENABLE_MODEL_CACHE: true,
    AGGRESSIVE_CLEANUP: true,
    WARMUP_ITERATIONS: 2        // Reducido de 3 a 2
  },

  // Rutas de modelos
  PATHS: {
    BASE_MODEL_URL: 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/5/default/1',
    CUSTOM_MODEL_PATH: './models/saved/plate_detector_model',
    WEIGHTS_PATH: './models/saved/plate_detector_weights',
    BACKUP_PATH: './models/backup/'
  },

  // Configuración de logging
  LOGGING: {
    ENABLE_PERFORMANCE_LOG: true,
    ENABLE_PREDICTION_LOG: false,
    LOG_LEVEL: 'info'
  },

  // Configuración de memoria
  MEMORY: {
    ENABLE_MEMORY_GROWTH: true,
    MEMORY_FRACTION: 0.3,
    CLEANUP_TENSORS: true
  }
};

// Configuraciones específicas por entorno
export const ENV_CONFIG = {
  development: {
    ...MODEL_CONFIG,
    LOGGING: {
      ...MODEL_CONFIG.LOGGING,
      ENABLE_PERFORMANCE_LOG: true,
      ENABLE_PREDICTION_LOG: true,
      LOG_LEVEL: 'debug'
    }
  },
  
  production: {
    ...MODEL_CONFIG,
    PERFORMANCE: {
      ...MODEL_CONFIG.PERFORMANCE,
      USE_GPU: true  // Habilitar GPU en producción si está disponible
    },
    LOGGING: {
      ...MODEL_CONFIG.LOGGING,
      ENABLE_PREDICTION_LOG: false,
      LOG_LEVEL: 'warn'
    }
  }
};

// Función para obtener configuración según entorno
export const getConfig = (env = 'development') => {
  return ENV_CONFIG[env] || MODEL_CONFIG;
};

// Validación de configuración
export const validateConfig = (config) => {
  const required = ['INPUT', 'MODEL', 'PREDICTION', 'PERFORMANCE'];
  
  for (const key of required) {
    if (!config[key]) {
      throw new Error(`Missing required config section: ${key}`);
    }
  }
  
  if (config.INPUT.WIDTH !== config.INPUT.HEIGHT) {
    throw new Error('Input dimensions must be square for MobileNet');
  }
  
  if (config.PERFORMANCE.TARGET_FPS < 1 || config.PERFORMANCE.TARGET_FPS > 10) {
    console.warn('TARGET_FPS outside recommended range (1-10)');
  }
  
  return true;
};

export default MODEL_CONFIG;
