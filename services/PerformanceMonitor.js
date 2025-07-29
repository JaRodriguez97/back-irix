/**
 * PerformanceMonitor - Servicio de Monitoreo de Performance en Tiempo Real
 * Analiza métricas de rendimiento y sugiere optimizaciones automáticas
 */

export class PerformanceMonitor {
  constructor() {
    this.metrics = {
      detections: [],
      memoryUsage: [],
      inferenceTime: [],
      tensorReuse: [],
      memoryLeaks: 0,
      startTime: Date.now()
    };
    
    this.thresholds = {
      SLOW_INFERENCE: 300,      // ms
      HIGH_MEMORY: 180 * 1024 * 1024, // 180MB
      LOW_TENSOR_REUSE: 0.3,    // 30%
      MAX_MEMORY_LEAKS: 5
    };
    
    this.recommendations = [];
    this.isMonitoring = false;
  }

  /**
   * Inicia el monitoreo de performance
   */
  startMonitoring() {
    if (this.isMonitoring) return;
    
    this.isMonitoring = true;
    this.metrics.startTime = Date.now();
    
    console.log('📊 PerformanceMonitor iniciado');
    
    // Monitoreo cada 30 segundos
    this.monitoringInterval = setInterval(() => {
      this._analyzePerformance();
    }, 30000);
  }

  /**
   * Detiene el monitoreo
   */
  stopMonitoring() {
    if (!this.isMonitoring) return;
    
    this.isMonitoring = false;
    
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    
    console.log('📊 PerformanceMonitor detenido');
  }

  /**
   * Registra una nueva detección
   * @param {Object} detectionStats - Estadísticas de la detección
   */
  recordDetection(detectionStats) {
    const record = {
      timestamp: Date.now(),
      processingTime: detectionStats.processingTime,
      memoryUsed: detectionStats.memoryUsed || 0,
      tensorReused: detectionStats.tensorReused || false,
      success: detectionStats.success || true
    };
    
    this.metrics.detections.push(record);
    this.metrics.inferenceTime.push(record.processingTime);
    this.metrics.memoryUsage.push(record.memoryUsed);
    
    if (record.tensorReused) {
      this.metrics.tensorReuse.push(1);
    } else {
      this.metrics.tensorReuse.push(0);
    }
    
    // Mantener solo los últimos 100 registros
    this._trimMetrics();
    
    // Análisis inmediato si hay problemas críticos
    if (record.processingTime > this.thresholds.SLOW_INFERENCE * 2) {
      this._analyzePerformance();
    }
  }

  /**
   * Registra un memory leak detectado
   */
  recordMemoryLeak() {
    this.metrics.memoryLeaks++;
    
    if (this.metrics.memoryLeaks > this.thresholds.MAX_MEMORY_LEAKS) {
      this._addRecommendation('CRITICAL', 'Memory leaks detectados frecuentemente. Reiniciar servicio recomendado.');
    }
  }

  /**
   * Obtiene estadísticas actuales
   * @returns {Object} Estadísticas completas
   */
  getStats() {
    const detectionCount = this.metrics.detections.length;
    const uptime = Date.now() - this.metrics.startTime;
    
    if (detectionCount === 0) {
      return {
        uptime: uptime,
        detectionCount: 0,
        averageInferenceTime: 0,
        averageMemoryUsage: 0,
        tensorReuseRate: 0,
        recommendations: this.recommendations
      };
    }
    
    return {
      uptime: uptime,
      detectionCount: detectionCount,
      averageInferenceTime: this._calculateAverage(this.metrics.inferenceTime),
      averageMemoryUsage: this._calculateAverage(this.metrics.memoryUsage),
      tensorReuseRate: this._calculateAverage(this.metrics.tensorReuse),
      memoryLeaks: this.metrics.memoryLeaks,
      recommendations: this.recommendations,
      performance: this._calculatePerformanceScore()
    };
  }

  /**
   * Obtiene recomendaciones de optimización
   * @returns {Array} Lista de recomendaciones
   */
  getRecommendations() {
    return this.recommendations;
  }

  /**
   * Limpia métricas y recomendaciones
   */
  reset() {
    this.metrics = {
      detections: [],
      memoryUsage: [],
      inferenceTime: [],
      tensorReuse: [],
      memoryLeaks: 0,
      startTime: Date.now()
    };
    
    this.recommendations = [];
    console.log('📊 PerformanceMonitor reset completado');
  }

  // Métodos privados

  _analyzePerformance() {
    this.recommendations = []; // Limpiar recomendaciones anteriores
    
    const stats = this.getStats();
    
    // Análisis de tiempo de inferencia
    if (stats.averageInferenceTime > this.thresholds.SLOW_INFERENCE) {
      this._addRecommendation('HIGH', 
        `Inferencia lenta: ${stats.averageInferenceTime}ms (target: <${this.thresholds.SLOW_INFERENCE}ms). Considerar optimización de modelo.`
      );
    }
    
    // Análisis de memoria
    if (stats.averageMemoryUsage > this.thresholds.HIGH_MEMORY) {
      this._addRecommendation('HIGH', 
        `Alto uso de memoria: ${(stats.averageMemoryUsage / 1024 / 1024).toFixed(2)}MB. Implementar limpieza más agresiva.`
      );
    }
    
    // Análisis de reutilización de tensores
    if (stats.tensorReuseRate < this.thresholds.LOW_TENSOR_REUSE) {
      this._addRecommendation('MEDIUM', 
        `Baja reutilización de tensores: ${(stats.tensorReuseRate * 100).toFixed(1)}%. Revisar tensor pooling.`
      );
    }
    
    // Análisis de memory leaks
    if (this.metrics.memoryLeaks > 0) {
      this._addRecommendation('MEDIUM', 
        `${this.metrics.memoryLeaks} memory leaks detectados. Verificar limpieza de tensores.`
      );
    }
    
    // Recomendaciones positivas
    if (stats.averageInferenceTime < 200 && stats.tensorReuseRate > 0.7) {
      this._addRecommendation('INFO', 
        'Performance óptima: Inferencia rápida y alta reutilización de tensores.'
      );
    }
    
    if (this.recommendations.length > 0) {
      console.log('📊 Análisis de performance completado:');
      this.recommendations.forEach(rec => {
        const emoji = rec.priority === 'CRITICAL' ? '🚨' : 
                     rec.priority === 'HIGH' ? '⚠️' : 
                     rec.priority === 'MEDIUM' ? '💡' : 'ℹ️';
        console.log(`${emoji} [${rec.priority}] ${rec.message}`);
      });
    }
  }

  _addRecommendation(priority, message) {
    this.recommendations.push({
      timestamp: Date.now(),
      priority: priority,
      message: message
    });
  }

  _calculateAverage(values) {
    if (values.length === 0) return 0;
    const sum = values.reduce((acc, val) => acc + val, 0);
    return Math.round(sum / values.length);
  }

  _calculatePerformanceScore() {
    const stats = this.getStats();
    
    if (stats.detectionCount === 0) return 0;
    
    let score = 100;
    
    // Penalizar inferencia lenta
    if (stats.averageInferenceTime > this.thresholds.SLOW_INFERENCE) {
      score -= Math.min(30, (stats.averageInferenceTime - this.thresholds.SLOW_INFERENCE) / 10);
    }
    
    // Penalizar alto uso de memoria
    if (stats.averageMemoryUsage > this.thresholds.HIGH_MEMORY) {
      score -= 20;
    }
    
    // Penalizar baja reutilización
    if (stats.tensorReuseRate < this.thresholds.LOW_TENSOR_REUSE) {
      score -= 15;
    }
    
    // Penalizar memory leaks
    score -= this.metrics.memoryLeaks * 5;
    
    return Math.max(0, Math.round(score));
  }

  _trimMetrics() {
    const maxRecords = 100;
    
    if (this.metrics.detections.length > maxRecords) {
      this.metrics.detections = this.metrics.detections.slice(-maxRecords);
    }
    
    if (this.metrics.inferenceTime.length > maxRecords) {
      this.metrics.inferenceTime = this.metrics.inferenceTime.slice(-maxRecords);
    }
    
    if (this.metrics.memoryUsage.length > maxRecords) {
      this.metrics.memoryUsage = this.metrics.memoryUsage.slice(-maxRecords);
    }
    
    if (this.metrics.tensorReuse.length > maxRecords) {
      this.metrics.tensorReuse = this.metrics.tensorReuse.slice(-maxRecords);
    }
  }
}

export default PerformanceMonitor;
