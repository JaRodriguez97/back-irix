/**
 * DetectionService - Servicio Principal de Detección de Placas Vehiculares
 * Orquesta todo el proceso de detección con manejo de errores y logging
 */

import PlateDetector from '../models/PlateDetector.js';
import { MODEL_CONFIG, getConfig } from '../models/modelConfig.js';
import { formatErrorResponse } from '../models/modelUtils.js';

export class DetectionService {
  constructor(environment = 'development') {
    this.config = getConfig(environment);
    this.detector = new PlateDetector(environment);
    this.isInitialized = false;
    
    // Estado del servicio
    this.serviceStats = {
      startTime: new Date().toISOString(),
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      avgResponseTime: 0,
      lastRequestTime: null,
      peakResponseTime: 0
    };

    // Cache de resultados (opcional)
    this.enableCache = false;
    this.resultCache = new Map();
    this.maxCacheSize = 100;
  }

  /**
   * Inicializa el servicio de detección
   */
  async initialize() {
    if (this.isInitialized) {
      return true;
    }

    try {
      console.log('🔧 Inicializando DetectionService...');

      // Inicializar detector de placas
      await this.detector.initialize();

      this.isInitialized = true;
      console.log('✅ DetectionService inicializado correctamente');
      
      // Log de configuración inicial
      this._logServiceConfig();

      return true;

    } catch (error) {
      console.error('❌ Error inicializando DetectionService:', error);
      throw new Error(`DetectionService initialization failed: ${error.message}`);
    }
  }

  /**
   * Detecta placa vehicular en imagen
   * @param {Buffer} imageBuffer - Buffer de imagen
   * @param {Object} options - Opciones adicionales
   * @returns {Promise<Object>} Resultado de detección
   */
  async detectPlate(imageBuffer, options = {}) {
    if (!this.isInitialized) {
      throw new Error('DetectionService not initialized. Call initialize() first.');
    }

    const requestId = this._generateRequestId();
    const startTime = Date.now();

    try {
      // Validar entrada
      if (!Buffer.isBuffer(imageBuffer)) {
        throw new Error('Invalid input: expected Buffer');
      }

      if (imageBuffer.length === 0) {
        throw new Error('Empty image buffer');
      }

      // Log inicio de procesamiento
      if (this.config.LOGGING.ENABLE_PERFORMANCE_LOG) {
        console.log(`🔍 [${requestId}] Iniciando detección - Tamaño: ${imageBuffer.length} bytes`);
      }

      // Verificar cache si está habilitado
      const cacheKey = options.enableCache ? this._generateCacheKey(imageBuffer) : null;
      if (cacheKey && this.resultCache.has(cacheKey)) {
        console.log(`📋 [${requestId}] Resultado desde cache`);
        const cachedResult = { ...this.resultCache.get(cacheKey) };
        cachedResult.fromCache = true;
        return cachedResult;
      }

      // Ejecutar detección
      const detectionResult = await this.detector.detectPlate(imageBuffer);

      // Procesar resultado
      const serviceResult = this._processDetectionResult(detectionResult, requestId, options);

      // Guardar en cache si está habilitado
      if (cacheKey && serviceResult.success) {
        this._updateCache(cacheKey, serviceResult);
      }

      // Actualizar estadísticas
      const responseTime = Date.now() - startTime;
      this._updateStats(responseTime, true);

      // Log resultado
      if (this.config.LOGGING.ENABLE_PERFORMANCE_LOG) {
        console.log(`✅ [${requestId}] Detección completada en ${responseTime}ms - Resultado: ${serviceResult.result.hasPlate ? 'PLACA' : 'SIN PLACA'}`);
      }

      return serviceResult;

    } catch (error) {
      const responseTime = Date.now() - startTime;
      this._updateStats(responseTime, false);

      console.error(`❌ [${requestId}] Error en detección:`, error);

      return this._createErrorResponse(error, requestId, responseTime);
    }
  }

  /**
   * Procesa múltiples imágenes en lote
   * @param {Buffer[]} imageBuffers - Array de buffers de imagen
   * @param {Object} options - Opciones de procesamiento
   * @returns {Promise<Object>} Resultado de procesamiento en lote
   */
  async detectPlatesInBatch(imageBuffers, options = {}) {
    if (!this.isInitialized) {
      throw new Error('DetectionService not initialized');
    }

    const batchId = this._generateRequestId();
    const startTime = Date.now();

    try {
      console.log(`📦 [${batchId}] Procesando lote de ${imageBuffers.length} imágenes`);

      const results = [];
      const batchStats = {
        total: imageBuffers.length,
        successful: 0,
        failed: 0,
        withPlates: 0,
        withoutPlates: 0
      };

      // Procesar secuencialmente (para evitar sobrecarga de memoria)
      for (let i = 0; i < imageBuffers.length; i++) {
        try {
          const result = await this.detectPlate(imageBuffers[i], {
            ...options,
            batchIndex: i,
            batchId
          });

          results.push(result);

          // Actualizar estadísticas del lote
          if (result.success) {
            batchStats.successful++;
            if (result.result.hasPlate) {
              batchStats.withPlates++;
            } else {
              batchStats.withoutPlates++;
            }
          } else {
            batchStats.failed++;
          }

        } catch (error) {
          results.push(this._createErrorResponse(error, `${batchId}-${i}`));
          batchStats.failed++;
        }
      }

      const totalTime = Date.now() - startTime;

      return {
        success: true,
        batchId,
        timestamp: new Date().toISOString(),
        stats: batchStats,
        results,
        performance: {
          totalTime,
          averageTimePerImage: Math.round(totalTime / imageBuffers.length),
          imagesPerSecond: Math.round((imageBuffers.length / totalTime) * 1000)
        }
      };

    } catch (error) {
      return {
        success: false,
        batchId,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Obtiene estadísticas del servicio
   * @returns {Object} Estadísticas completas
   */
  getServiceStats() {
    const detectorStats = this.detector.getStats();
    
    return {
      service: {
        ...this.serviceStats,
        uptime: Date.now() - new Date(this.serviceStats.startTime).getTime(),
        isInitialized: this.isInitialized,
        cacheEnabled: this.enableCache,
        cacheSize: this.resultCache.size
      },
      detector: detectorStats
    };
  }

  /**
   * Obtiene información de salud del servicio
   * @returns {Object} Estado de salud
   */
  getHealthStatus() {
    const stats = this.getServiceStats();
    const memoryUsage = process.memoryUsage();
    
    return {
      status: this.isInitialized ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      uptime: stats.service.uptime,
      detector: {
        ready: stats.detector.detector.isReady,
        modelLoaded: stats.detector.detector.modelLoaded
      },
      performance: {
        totalRequests: stats.service.totalRequests,
        successRate: stats.service.totalRequests > 0 
          ? Math.round((stats.service.successfulRequests / stats.service.totalRequests) * 100)
          : 0,
        avgResponseTime: stats.service.avgResponseTime
      },
      memory: {
        used: Math.round(memoryUsage.heapUsed / 1024 / 1024),
        total: Math.round(memoryUsage.heapTotal / 1024 / 1024),
        external: Math.round(memoryUsage.external / 1024 / 1024)
      }
    };
  }

  /**
   * Habilita/deshabilita cache de resultados
   * @param {boolean} enabled - Estado del cache
   * @param {number} maxSize - Tamaño máximo del cache
   */
  configureCache(enabled = true, maxSize = 100) {
    this.enableCache = enabled;
    this.maxCacheSize = maxSize;

    if (!enabled) {
      this.resultCache.clear();
    }

    console.log(`📋 Cache ${enabled ? 'habilitado' : 'deshabilitado'} (max: ${maxSize})`);
  }

  /**
   * Limpia cache de resultados
   */
  clearCache() {
    this.resultCache.clear();
    console.log('🗑️ Cache limpiado');
  }

  /**
   * Libera recursos del servicio
   */
  async dispose() {
    try {
      // Limpiar cache
      this.resultCache.clear();

      // Liberar detector
      await this.detector.dispose();

      this.isInitialized = false;
      console.log('🗑️ DetectionService liberado correctamente');

    } catch (error) {
      console.error('❌ Error liberando DetectionService:', error);
    }
  }

  // Métodos privados

  _processDetectionResult(detectionResult, requestId, options) {
    return {
      ...detectionResult,
      requestId,
      serviceVersion: '1.0.0',
      options: {
        fromCache: false,
        batchIndex: options.batchIndex || null,
        batchId: options.batchId || null
      }
    };
  }

  _createErrorResponse(error, requestId, responseTime = 0) {
    return {
      ...formatErrorResponse(error, responseTime),
      requestId,
      serviceVersion: '1.0.0'
    };
  }

  _updateStats(responseTime, success) {
    this.serviceStats.totalRequests++;
    this.serviceStats.lastRequestTime = new Date().toISOString();

    if (responseTime > this.serviceStats.peakResponseTime) {
      this.serviceStats.peakResponseTime = responseTime;
    }

    if (success) {
      this.serviceStats.successfulRequests++;
    } else {
      this.serviceStats.failedRequests++;
    }

    // Calcular tiempo promedio de respuesta
    this.serviceStats.avgResponseTime = Math.round(
      ((this.serviceStats.avgResponseTime * (this.serviceStats.totalRequests - 1)) + responseTime) / 
      this.serviceStats.totalRequests
    );
  }

  _generateRequestId() {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  _generateCacheKey(imageBuffer) {
    // Simple hash basado en tamaño y primeros bytes
    const size = imageBuffer.length;
    const sample = imageBuffer.slice(0, Math.min(100, size));
    return `cache_${size}_${sample.toString('base64').substr(0, 20)}`;
  }

  _updateCache(key, result) {
    // Limpiar cache si está lleno
    if (this.resultCache.size >= this.maxCacheSize) {
      const firstKey = this.resultCache.keys().next().value;
      this.resultCache.delete(firstKey);
    }

    this.resultCache.set(key, { ...result });
  }

  _logServiceConfig() {
    console.log('⚙️ Configuración del DetectionService:');
    console.log(`   🎯 Target FPS: ${this.config.PERFORMANCE.TARGET_FPS}`);
    console.log(`   ⏱️ Max Inference Time: ${this.config.PERFORMANCE.MAX_INFERENCE_TIME}ms`);
    console.log(`   🎚️ Confidence Threshold: ${this.config.PREDICTION.CONFIDENCE_THRESHOLD}`);
    console.log(`   📋 Cache: ${this.enableCache ? 'Habilitado' : 'Deshabilitado'}`);
    console.log(`   📊 Log Level: ${this.config.LOGGING.LOG_LEVEL}`);
  }
}

export default DetectionService;
