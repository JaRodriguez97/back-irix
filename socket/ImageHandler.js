/**
 * ImageHandler - Procesador de imágenes recibidas vía Socket.IO
 * Maneja la validación, conversión y procesamiento de imágenes en tiempo real
 */

import { DetectionService } from '../services/DetectionService.js';
import { IMAGE_CONFIG, ERROR_CODES } from './SocketEvents.js';
import sharp from 'sharp';

export class ImageHandler {
    constructor() {
        this.detectionService = null;
        this.isInitialized = false;
        
        // Estadísticas del handler
        this.stats = {
            totalImages: 0,
            successfulProcessing: 0,
            errors: 0,
            averageProcessingTime: 0,
            lastProcessTime: null
        };
    }

    /**
     * Inicializa el handler con el servicio de detección
     */
    async initialize() {
        if (this.isInitialized) {
            return true;
        }

        try {
            console.log('🖼️ Inicializando ImageHandler...');
            
            // Crear e inicializar servicio de detección
            this.detectionService = new DetectionService('production');
            await this.detectionService.initialize();
            
            this.isInitialized = true;
            console.log('✅ ImageHandler inicializado correctamente');
            
            return true;
        } catch (error) {
            console.error('❌ Error inicializando ImageHandler:', error);
            throw error;
        }
    }

    /**
     * Procesa una imagen recibida vía socket
     * @param {Object} imageData - Datos de la imagen del socket
     * @param {string} socketId - ID del socket cliente
     * @returns {Promise<Object>} Resultado del procesamiento
     */
    async processImageFromSocket(imageData, socketId) {
        const startTime = Date.now();
        
        try {
            if (!this.isInitialized) {
                throw new Error('ImageHandler not initialized');
            }

            // Validar datos de entrada
            const validationResult = this._validateImageData(imageData);
            if (!validationResult.isValid) {
                throw new Error(validationResult.error);
            }

            console.log(`🔍 [${socketId}] Procesando imagen - Tipo: ${imageData.format || 'unknown'}`);

            // Convertir imagen a buffer
            let imageBuffer;
            try {
                imageBuffer = await this._convertToBuffer(imageData);
            } catch (error) {
                throw new Error(`Error convirtiendo imagen: ${error.message}`);
            }

            // Validar y procesar imagen con Sharp
            let processedBuffer;
            try {
                processedBuffer = await this._preprocessImage(imageBuffer);
            } catch (error) {
                throw new Error(`Error procesando imagen: ${error.message}`);
            }

            // Ejecutar detección de placas
            const detectionResult = await this.detectionService.detectPlate(processedBuffer, {
                socketId,
                enableCache: false,
                realTime: true
            });

            // Actualizar estadísticas
            const processingTime = Date.now() - startTime;
            this._updateStats(processingTime, true);

            console.log(`✅ [${socketId}] Procesamiento completado en ${processingTime}ms`);

            return {
                success: true,
                socketId,
                timestamp: new Date().toISOString(),
                result: {
                    hasPlate: detectionResult.result?.hasPlate || false,
                    confidence: detectionResult.result?.confidence || 0,
                    processingTime
                },
                metadata: {
                    imageSize: imageBuffer.length,
                    processedSize: processedBuffer.length,
                    format: imageData.format || 'unknown'
                }
            };

        } catch (error) {
            const processingTime = Date.now() - startTime;
            this._updateStats(processingTime, false);

            console.error(`❌ [${socketId}] Error procesando imagen:`, error);

            return {
                success: false,
                socketId,
                timestamp: new Date().toISOString(),
                error: {
                    code: this._getErrorCode(error),
                    message: error.message,
                    processingTime
                }
            };
        }
    }

    /**
     * Valida los datos de imagen recibidos
     * @param {Object} imageData - Datos de la imagen
     * @returns {Object} Resultado de validación
     */
    _validateImageData(imageData) {
        // Verificar que existe data
        if (!imageData || !imageData.data) {
            return {
                isValid: false,
                error: 'No image data provided'
            };
        }

        // Verificar formato si se proporciona
        if (imageData.format && !IMAGE_CONFIG.ALLOWED_FORMATS.includes(imageData.format)) {
            return {
                isValid: false,
                error: `Unsupported format: ${imageData.format}`
            };
        }

        // Verificar tamaño si se proporciona
        if (imageData.size && imageData.size > IMAGE_CONFIG.MAX_SIZE) {
            return {
                isValid: false,
                error: `Image too large: ${imageData.size} bytes (max: ${IMAGE_CONFIG.MAX_SIZE})`
            };
        }

        return { isValid: true };
    }

    /**
     * Convierte los datos de imagen a Buffer
     * @param {Object} imageData - Datos de la imagen
     * @returns {Promise<Buffer>} Buffer de la imagen
     */
    async _convertToBuffer(imageData) {
        try {
            // Si ya es un buffer
            if (Buffer.isBuffer(imageData.data)) {
                return imageData.data;
            }

            // Si es base64
            if (typeof imageData.data === 'string') {
                // Remover prefijo data:image si existe
                const base64Data = imageData.data.replace(/^data:image\/[a-z]+;base64,/, '');
                return Buffer.from(base64Data, 'base64');
            }

            // Si es ArrayBuffer
            if (imageData.data instanceof ArrayBuffer) {
                return Buffer.from(imageData.data);
            }

            // Si es Uint8Array
            if (imageData.data instanceof Uint8Array) {
                return Buffer.from(imageData.data);
            }

            throw new Error('Unsupported image data format');

        } catch (error) {
            throw new Error(`Buffer conversion failed: ${error.message}`);
        }
    }

    /**
     * Preprocesa la imagen usando Sharp - Simplificado para WebP/JPEG
     * @param {Buffer} imageBuffer - Buffer de imagen original (ya viene 300x300 del frontend)
     * @returns {Promise<Buffer>} Buffer de imagen procesada
     */
    async _preprocessImage(imageBuffer) {
        try {
            const sharpInstance = sharp(imageBuffer);
            const metadata = await sharpInstance.metadata();
            
            console.log(`📥 Imagen recibida: ${metadata.format?.toUpperCase()} ${metadata.width}x${metadata.height}`);
            
            // El frontend ya envía 300x300, solo validamos dimensiones por seguridad
            const expectedSize = IMAGE_CONFIG.EXPECTED_DIMENSIONS.width;
            const needsResize = metadata.width !== expectedSize || metadata.height !== expectedSize;
            
            if (needsResize) {
                console.log(`⚠️ Redimensionando ${metadata.width}x${metadata.height} → ${expectedSize}x${expectedSize}`);
            }
            
            // Procesar según formato: WebP o JPEG (fallback)
            const processedBuffer = await sharpInstance
                .resize(expectedSize, expectedSize, {
                    fit: 'fill',
                    background: { r: 0, g: 0, b: 0, alpha: 1 },
                    withoutEnlargement: !needsResize // No redimensionar si ya es correcto
                })
                .jpeg({ 
                    quality: metadata.format === 'webp' ? 95 : 90, // Mejor calidad para WebP convertido
                    progressive: true
                })
                .toBuffer();

            console.log(`✅ Procesada: ${(processedBuffer.length/1024).toFixed(2)}KB`);
            return processedBuffer;

        } catch (error) {
            throw new Error(`Image preprocessing failed: ${error.message}`);
        }
    }

    /**
     * Determina el código de error apropiado
     * @param {Error} error - Error ocurrido
     * @returns {string} Código de error
     */
    _getErrorCode(error) {
        const message = error.message.toLowerCase();
        
        if (message.includes('not initialized')) {
            return ERROR_CODES.MODEL_NOT_LOADED;
        }
        if (message.includes('format') || message.includes('conversion')) {
            return ERROR_CODES.INVALID_FORMAT;
        }
        if (message.includes('too large') || message.includes('size')) {
            return ERROR_CODES.IMAGE_TOO_LARGE;
        }
        if (message.includes('timeout')) {
            return ERROR_CODES.PROCESSING_TIMEOUT;
        }
        
        return ERROR_CODES.INTERNAL_ERROR;
    }

    /**
     * Actualiza estadísticas del handler
     * @param {number} processingTime - Tiempo de procesamiento
     * @param {boolean} success - Si fue exitoso
     */
    _updateStats(processingTime, success) {
        this.stats.totalImages++;
        this.stats.lastProcessTime = processingTime;

        if (success) {
            this.stats.successfulProcessing++;
        } else {
            this.stats.errors++;
        }

        // Calcular promedio de tiempo de procesamiento
        const successfulCount = this.stats.successfulProcessing;
        if (successfulCount > 0) {
            this.stats.averageProcessingTime = Math.round(
                ((this.stats.averageProcessingTime * (successfulCount - 1)) + processingTime) / successfulCount
            );
        }
    }

    /**
     * Obtiene estadísticas del handler
     * @returns {Object} Estadísticas actuales
     */
    getStats() {
        return {
            ...this.stats,
            successRate: this.stats.totalImages > 0 ? 
                Math.round((this.stats.successfulProcessing / this.stats.totalImages) * 100) : 0,
            errorRate: this.stats.totalImages > 0 ? 
                Math.round((this.stats.errors / this.stats.totalImages) * 100) : 0
        };
    }

    /**
     * Reinicia estadísticas
     */
    resetStats() {
        this.stats = {
            totalImages: 0,
            successfulProcessing: 0,
            errors: 0,
            averageProcessingTime: 0,
            lastProcessTime: null
        };
    }
}

// Instancia singleton para reutilizar en toda la aplicación
export const imageHandler = new ImageHandler();
