/**
 * Configuration Management para detección de placas vehiculares
 * Principio: Centralización de configuraciones para fácil mantenimiento
 */

export class DetectionConfig {
    static get DEFAULT_CONFIG() {
        return {
            // Configuraciones de procesamiento de imágenes
            image: {
                thumbnailSize: { width: 300, height: 300 },
                maxHDSize: { width: 4096, height: 4096 },
                quality: 85,
                format: 'jpeg',
                compressionLevel: 6
            },
            
            // Configuraciones de TensorFlow
            tensorflow: {
                modelUrl: process.env.PLATE_MODEL_URL || 'https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1',
                inputSize: [300, 300, 3],
                scoreThreshold: 0.5,
                maxDetections: 10,
                nmsThreshold: 0.5
            },
            
            // Configuraciones de tiempo real (2fps)
            realTime: {
                fps: 2,
                intervalMs: 500,
                maxConcurrentProcessing: 4,
                bufferSize: 8,
                timeoutMs: 2000
            },
            
            // Configuraciones de WebSocket
            websocket: {
                chunkSize: 65536, // 64KB
                maxPayloadSize: 50 * 1024 * 1024, // 50MB para HD
                compressionThreshold: 1024,
                heartbeatInterval: 30000
            },
            
            // Configuraciones de memoria
            memory: {
                maxBufferSize: 100 * 1024 * 1024, // 100MB
                gcThreshold: 80, // % de uso de memoria
                poolSize: 10,
                reuseBuffers: true
            },
            
            // Configuraciones de Gemini
            gemini: {
                maxRetries: 3,
                timeoutMs: 15000,
                rateLimitDelay: 1000,
                batchSize: 1
            },
            
            // Configuraciones de rendimiento
            performance: {
                enableProfiling: process.env.NODE_ENV === 'development',
                logPerformance: true,
                warmupRuns: 3,
                cacheResults: true,
                cacheTTL: 300000 // 5 minutos
            }
        };
    }
    
    static get PLATE_DETECTION_LABELS() {
        return [
            'license_plate',
            'plate',
            'vehicle_plate',
            'car_plate',
            'motorcycle_plate'
        ];
    }
    
    static get SUPPORTED_FORMATS() {
        return ['jpeg', 'jpg', 'png', 'webp'];
    }
    
    static validate(config) {
        const required = ['image', 'tensorflow', 'realTime', 'websocket'];
        for (const key of required) {
            if (!config[key]) {
                throw new Error(`Missing required configuration: ${key}`);
            }
        }
        return true;
    }
    
    static merge(customConfig = {}) {
        const defaultConfig = this.DEFAULT_CONFIG;
        return this._deepMerge(defaultConfig, customConfig);
    }
    
    static _deepMerge(target, source) {
        const result = { ...target };
        
        for (const key in source) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                result[key] = this._deepMerge(result[key] || {}, source[key]);
            } else {
                result[key] = source[key];
            }
        }
        
        return result;
    }
}

export default DetectionConfig;
