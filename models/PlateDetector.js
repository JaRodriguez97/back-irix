/**
 * PlateDetector - Clase Principal del Modelo de Detección de Placas Vehiculares
 * Implementa MobileNet con transfer learning para clasificación binaria
 */

import TensorFlowAdapter from '../adapters/TensorFlowAdapter.js';
import { MODEL_CONFIG, getConfig, validateConfig } from './modelConfig.js';
import { 
  imageToTensor, 
  processPrediction, 
  validateImageInput,
  formatDetectionResponse,
  formatErrorResponse,
  cleanupTensors 
} from './modelUtils.js';

export class PlateDetector {
  constructor(environment = 'development') {
    this.config = getConfig(environment);
    validateConfig(this.config);
    
    this.tfAdapter = new TensorFlowAdapter(environment);
    this.model = null;
    this.isReady = false;
    this.isLoading = false;
    
    // Estadísticas de operación
    this.stats = {
      detectionsCount: 0,
      totalProcessingTime: 0,
      averageProcessingTime: 0,
      successfulDetections: 0,
      failedDetections: 0,
      lastDetectionTime: null
    };
  }

  /**
   * Inicializa el detector de placas
   * Carga o crea el modelo según configuración
   */
  async initialize() {
    if (this.isReady || this.isLoading) {
      return this._waitForReady();
    }

    this.isLoading = true;

    try {
      console.log('🚀 Inicializando PlateDetector...');

      // Inicializar adaptador de TensorFlow
      await this.tfAdapter.initialize();

      // Intentar cargar modelo preentrenado o crear uno nuevo
      this.model = await this._loadOrCreateModel();

      // Realizar warmup del modelo
      await this._warmupModel();

      this.isReady = true;
      this.isLoading = false;

      console.log('✅ PlateDetector inicializado correctamente');
      console.log(`📊 Configuración: ${this.config.MODEL.NAME} v${this.config.MODEL.VERSION}`);
      console.log(`🎯 Target FPS: ${this.config.PERFORMANCE.TARGET_FPS}`);

      return true;

    } catch (error) {
      this.isLoading = false;
      console.error('❌ Error inicializando PlateDetector:', error);
      throw new Error(`PlateDetector initialization failed: ${error.message}`);
    }
  }

  /**
   * Entrena el modelo con dataset de placas
   * @param {tf.data.Dataset} dataset - Dataset de entrenamiento
   * @param {tf.data.Dataset} valDataset - Dataset de validación
   * @returns {Promise<Object>} Estadísticas de entrenamiento
   */
  async train(dataset, valDataset) {
    try {
      console.log('📚 Iniciando entrenamiento del clasificador de placas...');
      const history = await this.model.fitDataset(dataset, {
        epochs: this.config.TRANSFER_LEARNING.EPOCHS || 10,
        validationData: valDataset,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(`🕒 Epoch ${epoch + 1}: loss = ${logs.loss}, val_loss = ${logs.val_loss}`);
          }
        }
      });

      console.log('✅ Entrenamiento completado');
      return history;

    } catch (error) {
      console.error('❌ Error en entrenamiento:', error);
      throw new Error(`Training failed: ${error.message}`);
    }
  }

  /**
   * Detecta presencia de placa vehicular en imagen
   * @param {Buffer} imageBuffer - Buffer de imagen de entrada
   * @returns {Promise<Object>} Resultado de detección formateado
   */
  async detectPlate(imageBuffer) {
    if (!this.isReady) {
      throw new Error('Detector not ready. Call initialize() first.');
    }

    const startTime = Date.now();
    let inputTensor = null;
    let prediction = null;

    try {
      // Validar imagen de entrada
      const imageValidation = await validateImageInput(imageBuffer);
      if (!imageValidation.isValid) {
        throw new Error(`Invalid image: ${imageValidation.error}`);
      }

      if (this.config.LOGGING.ENABLE_PREDICTION_LOG) {
        console.log(`📸 Procesando imagen: ${imageValidation.width}x${imageValidation.height} (${imageValidation.format})`);
      }

      // Convertir imagen a tensor
      inputTensor = await imageToTensor(imageBuffer);

      // Ejecutar inferencia
      prediction = await this.tfAdapter.predict(inputTensor, this.model);

      // Procesar resultado de predicción
      const predictionResult = await processPrediction(prediction);

      // Actualizar estadísticas
      const processingTime = Date.now() - startTime;
      this._updateStats(processingTime, true);

      // Formatear respuesta
      const response = formatDetectionResponse(predictionResult, processingTime);

      if (this.config.LOGGING.ENABLE_PREDICTION_LOG) {
        console.log(`🔍 Detección completada: ${predictionResult.hasPlate ? 'PLACA DETECTADA' : 'SIN PLACA'} (${predictionResult.probability})`);
      }

      return response;

    } catch (error) {
      const processingTime = Date.now() - startTime;
      this._updateStats(processingTime, false);

      console.error('❌ Error en detección:', error);
      return formatErrorResponse(error, processingTime);

    } finally {
      // Limpiar tensores para evitar memory leaks
      cleanupTensors([inputTensor, prediction]);
    }
  }

  /**
   * Procesa múltiples imágenes en lote (futuro uso)
   * @param {Buffer[]} imageBuffers - Array de buffers de imagen
   * @returns {Promise<Object[]>} Array de resultados
   */
  async detectPlates(imageBuffers) {
    if (!Array.isArray(imageBuffers)) {
      throw new Error('Expected array of image buffers');
    }

    const results = [];
    
    for (const imageBuffer of imageBuffers) {
      try {
        const result = await this.detectPlate(imageBuffer);
        results.push(result);
      } catch (error) {
        results.push(formatErrorResponse(error));
      }
    }

    return results;
  }

  /**
   * Obtiene estadísticas del detector
   * @returns {Object} Estadísticas completas
   */
  getStats() {
    const tfStats = this.tfAdapter.getStats();
    
    return {
      detector: {
        isReady: this.isReady,
        modelLoaded: this.model !== null,
        ...this.stats
      },
      tensorflow: tfStats,
      config: {
        environment: this.config.LOGGING.LOG_LEVEL,
        targetFPS: this.config.PERFORMANCE.TARGET_FPS,
        maxInferenceTime: this.config.PERFORMANCE.MAX_INFERENCE_TIME,
        confidenceThreshold: this.config.PREDICTION.CONFIDENCE_THRESHOLD
      }
    };
  }

  /**
   * Obtiene información del modelo cargado
   * @returns {Object} Información del modelo
   */
  getModelInfo() {
    if (!this.model) {
      return { loaded: false };
    }

    return {
      loaded: true,
      name: this.config.MODEL.NAME,
      version: this.config.MODEL.VERSION,
      inputShape: this.model.inputs[0].shape,
      outputShape: this.model.outputs[0].shape,
      layers: this.model.layers.length,
      trainableParams: this.model.countParams(),
      isCompiled: this.model.optimizer !== null
    };
  }

  /**
   * Actualiza configuración en tiempo de ejecución
   * @param {Object} newConfig - Nueva configuración parcial
   */
  updateConfig(newConfig) {
    if (!this.isReady) {
      throw new Error('Cannot update config before initialization');
    }

    // Fusionar configuración
    this.config = { ...this.config, ...newConfig };
    validateConfig(this.config);

    console.log('⚙️ Configuración actualizada');
  }

  /**
   * Libera recursos del detector
   */
  async dispose() {
    try {
      // Liberar modelo
      if (this.model) {
        this.model.dispose();
        this.model = null;
      }

      // Liberar adaptador
      await this.tfAdapter.dispose();

      // Reset de estado
      this.isReady = false;
      this.isLoading = false;

      console.log('🗑️ PlateDetector liberado correctamente');

    } catch (error) {
      console.error('❌ Error liberando PlateDetector:', error);
    }
  }

  // Métodos privados

  async _loadOrCreateModel() {
    try {
      // Intentar cargar modelo personalizado primero
      if (this.config.PATHS.CUSTOM_MODEL_PATH) {
        console.log('📥 Intentando cargar modelo personalizado...');
        return await this.tfAdapter.loadModel(this.config.PATHS.CUSTOM_MODEL_PATH);
      }
    } catch (error) {
      console.warn('⚠️ Modelo personalizado no disponible, creando modelo base');
    }

    try {
      // Intentar cargar desde URL
      if (this.config.PATHS.BASE_MODEL_URL) {
        console.log('🌐 Intentando cargar modelo desde URL...');
        return await this.tfAdapter.loadModel(this.config.PATHS.BASE_MODEL_URL);
      }
    } catch (error) {
      console.warn('⚠️ Modelo remoto no disponible, creando modelo base');
    }

    // Crear modelo base como fallback
    console.log('🏗️ Creando modelo MobileNet base...');
    return await this.tfAdapter.createMobileNetBase();
  }

  async _warmupModel() {
    console.log('🔥 Realizando warmup del modelo...');

    const warmupRuns = 3;
    const dummyImage = Buffer.alloc(300 * 300 * 3, 128); // Imagen gris 300x300

    for (let i = 0; i < warmupRuns; i++) {
      try {
        const tensor = await imageToTensor(dummyImage);
        const prediction = await this.tfAdapter.predict(tensor, this.model);
        
        // Limpiar inmediatamente
        cleanupTensors([tensor, prediction]);
        
      } catch (error) {
        console.warn(`⚠️ Error en warmup run ${i + 1}:`, error.message);
      }
    }

    console.log('✅ Warmup completado');
  }

  _updateStats(processingTime, success) {
    this.stats.detectionsCount++;
    this.stats.totalProcessingTime += processingTime;
    this.stats.averageProcessingTime = Math.round(
      this.stats.totalProcessingTime / this.stats.detectionsCount
    );
    this.stats.lastDetectionTime = new Date().toISOString();

    if (success) {
      this.stats.successfulDetections++;
    } else {
      this.stats.failedDetections++;
    }
  }

  async _waitForReady() {
    while (this.isLoading) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return this.isReady;
  }
}

export default PlateDetector;
