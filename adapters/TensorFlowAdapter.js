/**
 * TensorFlow.js Adapter para Detección de Placas Vehiculares
 * Wrapper para TensorFlow.js con optimizaciones y manejo de memoria
 */

import * as tf from '@tensorflow/tfjs';
import { MODEL_CONFIG, getConfig } from '../models/modelConfig.js';
import OptimizedMobileNet from '../models/OptimizedMobileNet.js';

export class TensorFlowAdapter {
  constructor(environment = 'development') {
    this.config = getConfig(environment);
    this.isInitialized = false;
    this.backend = null;
    this.memoryStats = {
      initial: null,
      current: null,
      peak: null
    };
    
    // 🚀 OPTIMIZACIÓN: Tensor Pool para reutilización
    this.tensorPool = {
      inputTensors: [],
      outputTensors: [],
      maxPoolSize: 5
    };
    
    // 🚀 OPTIMIZACIÓN: Model Cache
    this.modelCache = new Map();
    
    // 🚀 OPTIMIZACIÓN: Performance Metrics
    this.performanceMetrics = {
      inferenceCount: 0,
      totalInferenceTime: 0,
      averageInferenceTime: 0,
      memoryLeaksDetected: 0,
      tensorsReused: 0
    };
  }

  /**
   * Inicializa TensorFlow.js con configuraciones optimizadas
   */
  async initialize() {
    if (this.isInitialized) {
      return true;
    }

    try {
      console.log('🔧 Inicializando TensorFlow.js adapter...');

      // Configurar backend óptimo
      await this._setupBackend();

      // Aplicar optimizaciones de memoria
      this._configureMemoryOptimizations();

      // Aplicar optimizaciones de performance
      this._configurePerformanceOptimizations();

      // Registrar estadísticas iniciales de memoria
      this.memoryStats.initial = tf.memory();

      this.isInitialized = true;
      console.log('✅ TensorFlow.js adapter inicializado correctamente');
      console.log(`📊 Backend activo: ${tf.getBackend()}`);
      console.log(`💾 Memoria inicial: ${this.memoryStats.initial.numBytes} bytes`);

      return true;

    } catch (error) {
      console.error('❌ Error inicializando TensorFlow.js:', error);
      throw new Error(`TensorFlow initialization failed: ${error.message}`);
    }
  }

  /**
   * Carga modelo desde URL o path local
   * @param {string} modelPath - URL o path del modelo
   * @returns {Promise<tf.LayersModel>} Modelo cargado
   */
  async loadModel(modelPath) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      console.log(`📥 Cargando modelo desde: ${modelPath}`);
      
      const startTime = Date.now();
      const model = await tf.loadLayersModel(modelPath);
      const loadTime = Date.now() - startTime;

      console.log(`✅ Modelo cargado en ${loadTime}ms`);
      console.log(`🏗️ Modelo cargado correctamente`);

      return model;

    } catch (error) {
      console.error('❌ Error cargando modelo:', error);
      throw new Error(`Model loading failed: ${error.message}`);
    }
  }

  /**
   * 🚀 NUEVO: Crea modelo MobileNetV2 optimizado específico para placas
   * @returns {Promise<tf.LayersModel>} Modelo optimizado con mejores prestaciones
   */
  async createOptimizedMobileNet() {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      console.log('🚀 Creando modelo MobileNetV2 optimizado...');
      
      const optimizedMobileNet = new OptimizedMobileNet(this.config);
      const model = await optimizedMobileNet.createOptimizedModel();
      
      // Cachear el modelo optimizado
      this.modelCache.set('optimized_mobilenet', {
        model: model,
        optimizer: optimizedMobileNet,
        createdAt: Date.now(),
        info: optimizedMobileNet.getModelInfo()
      });
      
      console.log('✅ Modelo MobileNetV2 optimizado creado exitosamente');
      console.log('📊 Información del modelo:', optimizedMobileNet.getModelInfo());
      
      return model;
      
    } catch (error) {
      console.error('❌ Error creando modelo optimizado:', error);
      console.warn('⚠️ Fallback al modelo básico temporal...');
      
      // Fallback al modelo básico
      return await this.createMobileNetBase();
    }
  }

  /**
   * Carga modelo preentrenado de placas desde URL
   * @returns {Promise<tf.LayersModel>} Modelo preentrenado listo para usar
   */
  async createMobileNetBase() {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      console.log('🏗️ Cargando modelo preentrenado de placas...');

      // Intentar cargar modelo personalizado primero
      if (this.config.PATHS.CUSTOM_MODEL_PATH) {
        try {
          console.log('📥 Cargando modelo personalizado de placas...');
          const model = await tf.loadLayersModel(`file://${this.config.PATHS.CUSTOM_MODEL_PATH}/model.json`);
          console.log('✅ Modelo personalizado de placas cargado exitosamente');
          return model;
        } catch (error) {
          console.warn('⚠️ Modelo personalizado no disponible, usando fallback');
        }
      }

      // Fallback: Crear modelo básico para demo (TEMPORAL)
      console.log('🔧 Creando modelo básico temporal...');
      
      const input = tf.input({ shape: [224, 224, 3] });
      
      // Capas básicas para demo
      const conv1 = tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        inputShape: [224, 224, 3]
      }).apply(input);
      
      const pool1 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(conv1);
      const flatten = tf.layers.flatten().apply(pool1);
      const dense = tf.layers.dense({ units: 64, activation: 'relu' }).apply(flatten);
      const output = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(dense);
      
      const model = tf.model({ inputs: input, outputs: output });
      
      model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
      });

      console.log('✅ Modelo básico temporal creado (REEMPLAZAR con modelo preentrenado real)');
      console.log('📊 Arquitectura del modelo:');
      model.summary();

      return model;

    } catch (error) {
      console.error('❌ Error cargando modelo:', error);
      throw new Error(`Model loading failed: ${error.message}`);
    }
  }

  /**
   * Ejecuta inferencia optimizada con Tensor Pool
   * @param {tf.Tensor} inputTensor - Tensor de entrada
   * @param {tf.LayersModel} model - Modelo para inferencia
   * @returns {Promise<tf.Tensor>} Resultado de la predicción
   */
  async predict(inputTensor, model) {
    if (!this.isInitialized) {
      throw new Error('TensorFlow adapter not initialized');
    }

    const startTime = Date.now();
    let prediction = null;
    let reusedTensor = false;

    try {
      // 🚀 OPTIMIZACIÓN: Intentar reutilizar tensor del pool
      const pooledTensor = this._getTensorFromPool('input', inputTensor.shape);
      let processedInput = inputTensor;
      
      if (pooledTensor) {
        // Copiar datos al tensor reutilizado
        pooledTensor.assign(inputTensor);
        processedInput = pooledTensor;
        reusedTensor = true;
        this.performanceMetrics.tensorsReused++;
      }

      // Ejecutar predicción
      prediction = model.predict(processedInput);

      // 🚀 OPTIMIZACIÓN: Actualizar métricas de performance
      const inferenceTime = Date.now() - startTime;
      this._updatePerformanceMetrics(inferenceTime);
      this._updateMemoryStats();

      if (this.config.LOGGING.ENABLE_PERFORMANCE_LOG) {
        console.log(`⚡ Inferencia completada en ${inferenceTime}ms (Tensor reusado: ${reusedTensor})`);
        console.log(`💾 Memoria actual: ${this.memoryStats.current.numBytes} bytes`);
        console.log(`📊 Promedio inferencia: ${this.performanceMetrics.averageInferenceTime}ms`);
      }

      // Validar tiempo de inferencia
      if (inferenceTime > this.config.PERFORMANCE.MAX_INFERENCE_TIME) {
        console.warn(`⚠️ Inferencia lenta: ${inferenceTime}ms (target: ${this.config.PERFORMANCE.MAX_INFERENCE_TIME}ms)`);
      }

      // 🚀 OPTIMIZACIÓN: Devolver tensor al pool para reutilización
      if (reusedTensor) {
        this._returnTensorToPool('input', processedInput);
      }

      // ⚡ OPTIMIZACIÓN CRÍTICA: Limpieza selectiva de memoria
      this._optimizedMemoryCleanup();

      return prediction;

    } catch (error) {
      console.error('❌ Error en inferencia:', error);
      
      // Limpiar recursos en caso de error
      if (prediction) prediction.dispose();
      
      throw new Error(`Inference failed: ${error.message}`);
    }
  }

  /**
   * Obtiene estadísticas de memoria y performance
   * @returns {Object} Estadísticas del adapter
   */
  getStats() {
    this._updateMemoryStats();

    return {
      isInitialized: this.isInitialized,
      backend: tf.getBackend(),
      memory: {
        current: this.memoryStats.current,
        initial: this.memoryStats.initial,
        peak: this.memoryStats.peak,
        leakDetected: this._detectMemoryLeak()
      },
      config: {
        useGPU: this.config.PERFORMANCE.USE_GPU,
        maxInferenceTime: this.config.PERFORMANCE.MAX_INFERENCE_TIME,
        targetFPS: this.config.PERFORMANCE.TARGET_FPS
      }
    };
  }

  /**
   * Limpia recursos y libera memoria
   */
  async dispose() {
    try {
      // Ejecutar garbage collection de tensores
      if (this.config.MEMORY.CLEANUP_TENSORS) {
        tf.dispose();
      }

      this.isInitialized = false;
      console.log('🗑️ TensorFlow adapter limpiado correctamente');

    } catch (error) {
      console.error('❌ Error limpiando adapter:', error);
    }
  }

  // Métodos privados

  async _setupBackend() {
    // Intentar usar GPU si está habilitado
    if (this.config.PERFORMANCE.USE_GPU) {
      try {
        await tf.setBackend('tensorflow');
        console.log('🚀 Backend GPU configurado');
      } catch (error) {
        console.warn('⚠️ GPU no disponible, usando CPU');
        await tf.setBackend('cpu');
      }
    } else {
      await tf.setBackend('cpu');
      console.log('🖥️ Backend CPU configurado');
    }

    await tf.ready();
    this.backend = tf.getBackend();
  }

  _configureMemoryOptimizations() {
    const currentBackend = tf.getBackend();
    
    // Solo aplicar configuraciones WebGL si el backend lo soporta
    if (currentBackend === 'webgl') {
      try {
        // Configurar crecimiento de memoria
        if (this.config.MEMORY.ENABLE_MEMORY_GROWTH) {
          tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
          tf.env().set('WEBGL_FLUSH_THRESHOLD', -1);
        }

        // Configurar fracción de memoria
        if (this.config.MEMORY.MEMORY_FRACTION < 1.0) {
          tf.env().set('WEBGL_MEMORY_FRACTION', this.config.MEMORY.MEMORY_FRACTION);
        }
        
        console.log('🔧 Configuraciones WebGL aplicadas');
      } catch (error) {
        console.warn('⚠️ Error aplicando configuraciones WebGL:', error.message);
      }
    } else {
      console.log(`🖥️ Backend ${currentBackend} - omitiendo configuraciones WebGL`);
      // Configuraciones específicas para CPU si es necesario
      // (Por ahora, TensorFlow.js CPU no requiere configuraciones especiales)
    }
  }

  _configurePerformanceOptimizations() {
    const currentBackend = tf.getBackend();
    
    // Solo aplicar configuraciones WebGL si el backend lo soporta
    if (currentBackend === 'webgl') {
      try {
        // Optimizaciones específicas para WebGL
        tf.env().set('WEBGL_PACK', true);
        tf.env().set('WEBGL_FORCE_F16_TEXTURES', false); // Mejor para accuracy
        tf.env().set('WEBGL_EXP_CONV', true);
        
        // Optimizar para batch size pequeño
        tf.env().set('WEBGL_MAX_TEXTURE_SIZE', 4096);
        
        console.log('⚡ Optimizaciones WebGL aplicadas');
      } catch (error) {
        console.warn('⚠️ Error aplicando optimizaciones WebGL:', error.message);
      }
    } else {
      console.log(`💾 Backend ${currentBackend} - usando configuraciones básicas`);
      // Para CPU no necesitamos configuraciones especiales de TensorFlow.js
      // Las optimizaciones de CPU son automáticas
    }
  }

  _validateModelInput(model) {
    const inputShape = model.inputs[0].shape;
    const expectedShape = [
      null, // batch dimension
      this.config.INPUT.HEIGHT,
      this.config.INPUT.WIDTH,
      this.config.INPUT.CHANNELS
    ];

    if (inputShape.length !== expectedShape.length) {
      throw new Error(`Invalid model input shape: expected ${expectedShape}, got ${inputShape}`);
    }

    for (let i = 1; i < expectedShape.length; i++) {
      if (inputShape[i] !== expectedShape[i]) {
        throw new Error(`Invalid model input dimension ${i}: expected ${expectedShape[i]}, got ${inputShape[i]}`);
      }
    }
  }

  _updateMemoryStats() {
    this.memoryStats.current = tf.memory();
    
    if (!this.memoryStats.peak || 
        this.memoryStats.current.numBytes > this.memoryStats.peak.numBytes) {
      this.memoryStats.peak = { ...this.memoryStats.current };
    }
  }

  _detectMemoryLeak() {
    if (!this.memoryStats.initial || !this.memoryStats.current) {
      return false;
    }

    const memoryIncrease = this.memoryStats.current.numBytes - this.memoryStats.initial.numBytes;
    const maxMemoryMB = this.config.PERFORMANCE.MAX_MEMORY_MB * 1024 * 1024;

    return memoryIncrease > maxMemoryMB;
  }

  /**
   * Fuerza limpieza inmediata de tensores para evitar memory leaks
   * @private
   */
  _forceTensorCleanup() {
    try {
      // Ejecutar garbage collection de tensores no utilizados
      const memoryBefore = tf.memory();
      
      // Forzar limpieza de tensores
      tf.dispose();
      
      // Actualizar estadísticas de memoria
      this._updateMemoryStats();
      
      const memoryAfter = tf.memory();
      const memoryFreed = memoryBefore.numBytes - memoryAfter.numBytes;
      
      if (this.config.LOGGING.ENABLE_PERFORMANCE_LOG && memoryFreed > 0) {
        console.log(`🧧 Memoria liberada: ${(memoryFreed / 1024).toFixed(2)}KB`);
      }
      
    } catch (error) {
      console.warn('⚠️ Error en limpieza de tensores:', error.message);
    }
  }

  /**
   * 🚀 OPTIMIZACIÓN: Obtiene tensor del pool o crea uno nuevo
   * @private
   */
  _getTensorFromPool(type, shape) {
    const pool = type === 'input' ? this.tensorPool.inputTensors : this.tensorPool.outputTensors;
    
    // Buscar tensor compatible en el pool
    for (let i = 0; i < pool.length; i++) {
      const tensor = pool[i];
      if (this._shapeEquals(tensor.shape, shape)) {
        // Remover del pool y retornar
        pool.splice(i, 1);
        return tensor;
      }
    }
    
    return null; // No hay tensor compatible
  }

  /**
   * 🚀 OPTIMIZACIÓN: Devuelve tensor al pool para reutilización
   * @private
   */
  _returnTensorToPool(type, tensor) {
    const pool = type === 'input' ? this.tensorPool.inputTensors : this.tensorPool.outputTensors;
    
    // Solo agregar si el pool no está lleno
    if (pool.length < this.tensorPool.maxPoolSize) {
      pool.push(tensor);
    } else {
      // Pool lleno, liberar tensor
      tensor.dispose();
    }
  }

  /**
   * 🚀 OPTIMIZACIÓN: Limpieza optimizada de memoria
   * @private
   */
  _optimizedMemoryCleanup() {
    try {
      const memoryBefore = tf.memory();
      
      // Solo limpiar si hay muchos tensores
      if (memoryBefore.numTensors > 50) {
        tf.dispose();
        
        const memoryAfter = tf.memory();
        const memoryFreed = memoryBefore.numBytes - memoryAfter.numBytes;
        
        if (this.config.LOGGING.ENABLE_PERFORMANCE_LOG && memoryFreed > 1024) {
          console.log(`🧧 Limpieza optimizada: ${(memoryFreed / 1024).toFixed(2)}KB liberados`);
        }
      }
      
      this._updateMemoryStats();
      
    } catch (error) {
      console.warn('⚠️ Error en limpieza optimizada:', error.message);
    }
  }

  /**
   * 🚀 OPTIMIZACIÓN: Actualiza métricas de performance
   * @private
   */
  _updatePerformanceMetrics(inferenceTime) {
    this.performanceMetrics.inferenceCount++;
    this.performanceMetrics.totalInferenceTime += inferenceTime;
    this.performanceMetrics.averageInferenceTime = Math.round(
      this.performanceMetrics.totalInferenceTime / this.performanceMetrics.inferenceCount
    );
    
    // Detectar memory leaks
    if (this._detectMemoryLeak()) {
      this.performanceMetrics.memoryLeaksDetected++;
    }
  }

  /**
   * Compara si dos shapes son iguales
   * @private
   */
  _shapeEquals(shape1, shape2) {
    if (shape1.length !== shape2.length) return false;
    
    for (let i = 0; i < shape1.length; i++) {
      if (shape1[i] !== shape2[i]) return false;
    }
    
    return true;
  }

  /**
   * 🚀 OPTIMIZACIÓN: Obtiene estadísticas completas incluyendo optimizaciones
   * @returns {Object} Estadísticas completas
   */
  getCompleteStats() {
    this._updateMemoryStats();

    return {
      ...this.getStats(),
      performance: {
        ...this.performanceMetrics,
        tensorPoolSize: {
          input: this.tensorPool.inputTensors.length,
          output: this.tensorPool.outputTensors.length,
          maxSize: this.tensorPool.maxPoolSize
        },
        modelCacheSize: this.modelCache.size,
        efficiency: {
          reuseRate: this.performanceMetrics.inferenceCount > 0 
            ? (this.performanceMetrics.tensorsReused / this.performanceMetrics.inferenceCount * 100).toFixed(2) + '%'
            : '0%',
          memoryLeakRate: this.performanceMetrics.inferenceCount > 0
            ? (this.performanceMetrics.memoryLeaksDetected / this.performanceMetrics.inferenceCount * 100).toFixed(2) + '%'
            : '0%'
        }
      }
    };
  }
}

export default TensorFlowAdapter;
