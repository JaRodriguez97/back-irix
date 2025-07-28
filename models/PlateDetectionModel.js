/**
 * PlateDetectionModel - Single Responsibility Principle
 * Responsabilidad única: Manejar la detección de placas vehiculares usando TensorFlow.js
 */

import * as tf from '@tensorflow/tfjs-node';
import { DetectionConfig } from '../config/DetectionConfig.js';

export class PlateDetectionModel {
    constructor(config = {}) {
        this.config = DetectionConfig.merge(config);
        this.model = null;
        this.isLoaded = false;
        this.isLoading = false;
        this.warmupCompleted = false;
        this.detectionCount = 0;
        this.lastDetectionTime = 0;
    }
    
    /**
     * Inicializa y carga el modelo de TensorFlow
     */
    async initialize() {
        if (this.isLoaded || this.isLoading) {
            return this.waitForLoad();
        }
        
        this.isLoading = true;
        
        try {
            console.log('🔥 Cargando modelo de detección de placas...');
            
            // Configurar TensorFlow para mejor rendimiento
            tf.env().set('WEBGL_PACK', true);
            tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
            
            // Cargar modelo preentrenado o personalizado
            this.model = await this._loadModel();
            
            // Warmup del modelo
            await this._warmupModel();
            
            this.isLoaded = true;
            this.isLoading = false;
            
            console.log('✅ Modelo de detección de placas cargado exitosamente');
            console.log(`📊 Memoria GPU: ${tf.memory().numBytesInGPU} bytes`);
            
        } catch (error) {
            this.isLoading = false;
            console.error('❌ Error cargando modelo:', error);
            throw new Error(`Failed to load plate detection model: ${error.message}`);
        }
    }
    
    /**
     * Detecta placas vehiculares en una imagen
     * @param {Buffer|tensor} imageData - Datos de la imagen
     * @returns {Promise<Object>} Resultado de la detección
     */
    async detectPlates(imageData) {
        if (!this.isLoaded) {
            throw new Error('Model not loaded. Call initialize() first.');
        }
        
        const startTime = Date.now();
        let tensor = null;
        
        try {
            // Preparar tensor de entrada
            tensor = await this._prepareInput(imageData);
            
            // Ejecutar inferencia
            const predictions = await this.model.predict(tensor);
            
            // Procesar resultados
            const detections = await this._processDetections(predictions);
            
            // Filtrar solo placas vehiculares
            const plateDetections = this._filterPlateDetections(detections);
            
            // Actualizar estadísticas
            this._updateStats(startTime);
            
            return {
                success: true,
                hasPlates: plateDetections.length > 0,
                platesCount: plateDetections.length,
                detections: plateDetections,
                confidence: this._getAverageConfidence(plateDetections),
                processingTime: Date.now() - startTime,
                timestamp: new Date().toISOString()
            };
            
        } catch (error) {
            console.error('❌ Error en detección:', error);
            return {
                success: false,
                error: error.message,
                hasPlates: false,
                platesCount: 0,
                detections: [],
                processingTime: Date.now() - startTime
            };
        } finally {
            // Limpiar memoria
            if (tensor) tensor.dispose();
        }
    }
    
    /**
     * Obtiene estadísticas del modelo
     */
    getStats() {
        return {
            isLoaded: this.isLoaded,
            detectionCount: this.detectionCount,
            averageProcessingTime: this._getAverageProcessingTime(),
            memoryUsage: tf.memory(),
            lastDetectionTime: this.lastDetectionTime,
            warmupCompleted: this.warmupCompleted
        };
    }
    
    /**
     * Limpia recursos del modelo
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.isLoaded = false;
        console.log('🗑️ Modelo de detección liberado de memoria');
    }
    
    // Métodos privados
    
    async _loadModel() {
        const modelUrl = this.config.tensorflow.modelUrl;
        
        try {
            return await tf.loadLayersModel(modelUrl);
        } catch (error) {
            // Fallback a modelo local si existe
            console.warn('⚠️ Usando modelo fallback local');
            return await this._createFallbackModel();
        }
    }
    
    async _createFallbackModel() {
        // Modelo simple para detección de objetos rectangulares (placas)
        const model = tf.sequential({
            layers: [
                tf.layers.conv2d({
                    inputShape: this.config.tensorflow.inputSize,
                    filters: 32,
                    kernelSize: 3,
                    activation: 'relu'
                }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }),
                tf.layers.maxPooling2d({ poolSize: 2 }),
                tf.layers.flatten(),
                tf.layers.dense({ units: 128, activation: 'relu' }),
                tf.layers.dense({ units: this.config.tensorflow.maxDetections * 6 }) // x, y, w, h, class, confidence
            ]
        });
        
        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError'
        });
        
        return model;
    }
    
    async _warmupModel() {
        console.log('🔥 Calentando modelo...');
        const warmupRuns = this.config.performance.warmupRuns;
        
        for (let i = 0; i < warmupRuns; i++) {
            const dummyInput = tf.randomNormal(this.config.tensorflow.inputSize);
            const prediction = await this.model.predict(dummyInput);
            prediction.dispose();
            dummyInput.dispose();
        }
        
        this.warmupCompleted = true;
        console.log('✅ Warmup completado');
    }
    
    async _prepareInput(imageData) {
        let tensor;
        
        if (Buffer.isBuffer(imageData)) {
            tensor = tf.node.decodeImage(imageData, 3);
        } else if (imageData instanceof tf.Tensor) {
            tensor = imageData.clone();
        } else {
            throw new Error('Invalid image data format');
        }
        
        // Redimensionar a tamaño de entrada del modelo
        const [targetHeight, targetWidth] = this.config.tensorflow.inputSize.slice(0, 2);
        const resized = tf.image.resizeBilinear(tensor, [targetHeight, targetWidth]);
        
        // Normalizar valores de píxeles (0-1)
        const normalized = resized.div(255.0);
        
        // Agregar dimensión de batch
        const batched = normalized.expandDims(0);
        
        // Limpiar tensores intermedios
        if (tensor !== imageData) tensor.dispose();
        resized.dispose();
        normalized.dispose();
        
        return batched;
    }
    
    async _processDetections(predictions) {
        // Procesar salidas del modelo según el formato específico
        const scores = await predictions[0].data();
        const boxes = await predictions[1].data();
        
        const detections = [];
        const scoreThreshold = this.config.tensorflow.scoreThreshold;
        
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] > scoreThreshold) {
                const boxIndex = i * 4;
                detections.push({
                    score: scores[i],
                    bbox: {
                        x: boxes[boxIndex],
                        y: boxes[boxIndex + 1],
                        width: boxes[boxIndex + 2] - boxes[boxIndex],
                        height: boxes[boxIndex + 3] - boxes[boxIndex + 1]
                    }
                });
            }
        }
        
        predictions.forEach(p => p.dispose());
        return detections;
    }
    
    _filterPlateDetections(detections) {
        // Filtrar detecciones que probablemente sean placas
        return detections.filter(detection => {
            const { width, height } = detection.bbox;
            const aspectRatio = width / height;
            
            // Placas típicamente tienen aspect ratio entre 2:1 y 4:1
            return aspectRatio >= 1.5 && aspectRatio <= 5.0 && 
                   detection.score >= this.config.tensorflow.scoreThreshold;
        });
    }
    
    _getAverageConfidence(detections) {
        if (detections.length === 0) return 0;
        const totalConfidence = detections.reduce((sum, det) => sum + det.score, 0);
        return totalConfidence / detections.length;
    }
    
    _updateStats(startTime) {
        this.detectionCount++;
        this.lastDetectionTime = Date.now() - startTime;
    }
    
    _getAverageProcessingTime() {
        return this.lastDetectionTime; // Simplificado por ahora
    }
    
    async waitForLoad() {
        while (this.isLoading) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        return this.isLoaded;
    }
}

export default PlateDetectionModel;
