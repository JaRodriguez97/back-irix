/**
 * Utilidades de Procesamiento para Detección de Placas Vehiculares
 * Funciones para conversión Sharp → Tensor y normalización
 */

import * as tf from '@tensorflow/tfjs';
import sharp from 'sharp';
import { MODEL_CONFIG } from './modelConfig.js';

/**
 * Convierte imagen Sharp a tensor normalizado para MobileNet
 * @param {Buffer} imageBuffer - Buffer de imagen de entrada
 * @returns {Promise<tf.Tensor>} Tensor procesado listo para predicción
 */
export const imageToTensor = async (imageBuffer) => {
  try {
    // Redimensionar imagen 300x300 → 224x224 usando Sharp
    const processedBuffer = await sharp(imageBuffer)
      .resize(MODEL_CONFIG.INPUT.WIDTH, MODEL_CONFIG.INPUT.HEIGHT, {
        fit: 'cover',           // Mantener aspecto, recortar si es necesario
        position: 'center'      // Centrar el recorte
      })
      .removeAlpha()            // Quitar canal alpha si existe
      .raw()                    // Obtener datos raw RGB
      .toBuffer();

    // Convertir buffer a tensor
    const tensor = tf.tensor3d(
      new Uint8Array(processedBuffer),
      [MODEL_CONFIG.INPUT.HEIGHT, MODEL_CONFIG.INPUT.WIDTH, MODEL_CONFIG.INPUT.CHANNELS],
      'int32'
    );

    // Normalizar píxeles 0-255 → 0-1 (estándar para modelos preentrenados)
    const normalized = tensor.cast('float32').div(255.0);

    // Agregar dimensión de batch [1, 224, 224, 3]
    const batched = normalized.expandDims(0);

    // Limpiar tensores intermedios
    tensor.dispose();
    normalized.dispose();

    return batched;

  } catch (error) {
    throw new Error(`Error processing image to tensor: ${error.message}`);
  }
};

/**
 * Aplica normalización ImageNet estándar
 * @param {tf.Tensor} tensor - Tensor normalizado 0-1
 * @returns {tf.Tensor} Tensor con normalización ImageNet
 */
const normalizeImageNet = (tensor) => {
  const mean = tf.tensor1d(MODEL_CONFIG.MODEL.MEAN);
  const std = tf.tensor1d(MODEL_CONFIG.MODEL.STD);
  
  const normalized = tensor.sub(mean).div(std);
  
  mean.dispose();
  std.dispose();
  
  return normalized;
};

/**
 * Procesa la salida del modelo para obtener probabilidad de placa
 * @param {tf.Tensor} prediction - Salida del modelo (sigmoid)
 * @returns {Promise<Object>} Resultado procesado
 */
export const processPrediction = async (prediction) => {
  try {
    // Obtener valor de probabilidad
    const probabilityArray = await prediction.data();
    const probability = probabilityArray[0]; // Primera (y única) salida

    // Determinar si hay placa basado en umbral
    const hasPlate = probability >= MODEL_CONFIG.PREDICTION.CONFIDENCE_THRESHOLD;

    // Calcular nivel de confianza
    const confidence = calculateConfidenceLevel(probability);

    return {
      probability: Math.round(probability * 1000) / 1000, // 3 decimales
      hasPlate,
      confidence,
      threshold: MODEL_CONFIG.PREDICTION.CONFIDENCE_THRESHOLD,
      raw: probability
    };

  } catch (error) {
    throw new Error(`Error processing prediction: ${error.message}`);
  }
};

/**
 * Calcula nivel de confianza basado en la probabilidad
 * @param {number} probability - Probabilidad 0-1
 * @returns {string} Nivel de confianza
 */
const calculateConfidenceLevel = (probability) => {
  if (probability < MODEL_CONFIG.PREDICTION.MIN_CONFIDENCE) return 'very_low';
  if (probability < 0.3) return 'low';
  if (probability < 0.7) return 'medium';
  if (probability < MODEL_CONFIG.PREDICTION.MAX_CONFIDENCE) return 'high';
  return 'very_high';
};

/**
 * Valida dimensiones de imagen de entrada
 * @param {Buffer} imageBuffer - Buffer de imagen
 * @returns {Promise<Object>} Información de la imagen
 */
export const validateImageInput = async (imageBuffer) => {
  try {
    const metadata = await sharp(imageBuffer).metadata();
    
    return {
      isValid: true,
      width: metadata.width,
      height: metadata.height,
      channels: metadata.channels,
      format: metadata.format,
      size: imageBuffer.length,
      needsResize: metadata.width !== MODEL_CONFIG.INPUT.ORIGINAL_SIZE || 
                   metadata.height !== MODEL_CONFIG.INPUT.ORIGINAL_SIZE
    };

  } catch (error) {
    return {
      isValid: false,
      error: error.message
    };
  }
};

/**
 * Prepara respuesta de detección formateada
 * @param {Object} predictionResult - Resultado de processPrediction
 * @param {number} processingTime - Tiempo de procesamiento en ms
 * @returns {Object} Respuesta formateada
 */
export const formatDetectionResponse = (predictionResult, processingTime) => {
  return {
    success: true,
    timestamp: new Date().toISOString(),
    result: {
      hasPlate: predictionResult.hasPlate,
      probability: predictionResult.probability,
      confidence: predictionResult.confidence,
      threshold: predictionResult.threshold
    },
    performance: {
      processingTime: Math.round(processingTime),
      targetTime: MODEL_CONFIG.PERFORMANCE.MAX_INFERENCE_TIME,
      withinTarget: processingTime <= MODEL_CONFIG.PERFORMANCE.MAX_INFERENCE_TIME
    },
    model: {
      version: MODEL_CONFIG.MODEL.VERSION,
      inputSize: `${MODEL_CONFIG.INPUT.WIDTH}x${MODEL_CONFIG.INPUT.HEIGHT}`
    }
  };
};

/**
 * Maneja errores de procesamiento y genera respuesta de error
 * @param {Error} error - Error ocurrido
 * @param {number} processingTime - Tiempo transcurrido
 * @returns {Object} Respuesta de error formateada
 */
export const formatErrorResponse = (error, processingTime = 0) => {
  return {
    success: false,
    timestamp: new Date().toISOString(),
    error: {
      message: error.message,
      type: error.name || 'ProcessingError'
    },
    result: {
      hasPlate: null,
      probability: null,
      confidence: 'error'
    },
    performance: {
      processingTime: Math.round(processingTime),
      failed: true
    }
  };
};

/**
 * Limpia tensor de manera segura
 * @param {tf.Tensor|tf.Tensor[]} tensors - Tensor o array de tensors
 */
export const cleanupTensors = (tensors) => {
  if (!tensors) return;
  
  if (Array.isArray(tensors)) {
    tensors.forEach(tensor => {
      if (tensor && typeof tensor.dispose === 'function') {
        tensor.dispose();
      }
    });
  } else if (typeof tensors.dispose === 'function') {
    tensors.dispose();
  }
};

export default {
  imageToTensor,
  processPrediction,
  validateImageInput,
  formatDetectionResponse,
  formatErrorResponse,
  cleanupTensors
};
