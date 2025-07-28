/**
 * TrainingService - Servicio para Entrenamiento del Modelo de Placas
 * Implementa transfer learning con MobileNetV2 y dataset personalizado
 */

import * as tf from '@tensorflow/tfjs';
import fs from 'fs/promises';
import path from 'path';
import sharp from 'sharp';
import PlateDetector from '../models/PlateDetector.js';
import { getConfig } from '../models/modelConfig.js';

export class TrainingService {
  constructor(environment = 'development') {
    this.config = getConfig(environment);
    this.plateDetector = new PlateDetector(environment);
    this.isTraining = false;
    this.trainingStats = {
      startTime: null,
      endTime: null,
      epochCount: 0,
      bestLoss: Infinity,
      bestAccuracy: 0
    };
  }

  /**
   * Prepara dataset de imágenes para entrenamiento
   * @param {string} datasetPath - Ruta al directorio del dataset
   * @returns {Promise<{trainData: tf.data.Dataset, valData: tf.data.Dataset}>}
   */
  async prepareDataset(datasetPath) {
    try {
      console.log('📂 Preparando dataset para entrenamiento...');
      
      // Buscar subdirectorios con imágenes etiquetadas (estructura simple)
      const siPath = path.join(datasetPath, 'si');     // Imágenes CON placas
      const noPath = path.join(datasetPath, 'no');     // Imágenes SIN placas
      
      // Validar estructura del dataset
      await this._validateDatasetStructure(siPath, noPath);
      
      // Cargar imágenes con placas (etiqueta: 1)
      const withPlatesImages = await this._loadImagesFromDirectory(siPath, 1);
      
      // Cargar imágenes sin placas (etiqueta: 0)
      const withoutPlatesImages = await this._loadImagesFromDirectory(noPath, 0);
      
      // Combinar y mezclar dataset
      const allImages = [...withPlatesImages, ...withoutPlatesImages];
      const shuffledImages = this._shuffleArray(allImages);
      
      console.log(`📊 Dataset preparado: ${withPlatesImages.length} con placas, ${withoutPlatesImages.length} sin placas`);
      
      // Dividir en entrenamiento y validación
      const splitIndex = Math.floor(shuffledImages.length * (1 - this.config.TRANSFER_LEARNING.VALIDATION_SPLIT));
      const trainImages = shuffledImages.slice(0, splitIndex);
      const valImages = shuffledImages.slice(splitIndex);
      
      // Crear datasets de TensorFlow
      const trainData = tf.data.array(trainImages)
        .map(({ image, label }) => ({ xs: image, ys: tf.scalar(label, 'float32') }))
        .batch(this.config.TRANSFER_LEARNING.BATCH_SIZE)
        .prefetch(1);
        
      const valData = tf.data.array(valImages)
        .map(({ image, label }) => ({ xs: image, ys: tf.scalar(label, 'float32') }))
        .batch(this.config.TRANSFER_LEARNING.BATCH_SIZE)
        .prefetch(1);
      
      console.log(`✅ Dataset dividido: ${trainImages.length} entrenamiento, ${valImages.length} validación`);
      
      return { trainData, valData };
      
    } catch (error) {
      console.error('❌ Error preparando dataset:', error);
      throw new Error(`Dataset preparation failed: ${error.message}`);
    }
  }

  /**
   * Entrena el modelo con el dataset preparado
   * @param {string} datasetPath - Ruta al dataset
   * @param {Object} options - Opciones de entrenamiento
   * @returns {Promise<Object>} Resultados del entrenamiento
   */
  async trainModel(datasetPath, options = {}) {
    if (this.isTraining) {
      throw new Error('Training already in progress');
    }

    this.isTraining = true;
    this.trainingStats.startTime = new Date();

    try {
      console.log('🚀 Iniciando proceso de entrenamiento...');
      
      // Inicializar detector y modelo
      await this.plateDetector.initialize();
      
      // Preparar dataset
      const { trainData, valData } = await this.prepareDataset(datasetPath);
      
      // Configurar callbacks de entrenamiento
      const callbacks = this._createTrainingCallbacks();
      
      // Entrenar modelo
      console.log('📚 Entrenando clasificador de placas...');
      const history = await this.plateDetector.train(trainData, valData);
      
      // Guardar modelo entrenado
      await this._saveTrainedModel();
      
      this.trainingStats.endTime = new Date();
      const trainingTime = this.trainingStats.endTime - this.trainingStats.startTime;
      
      console.log(`✅ Entrenamiento completado en ${Math.round(trainingTime / 1000)}s`);
      
      const results = {
        success: true,
        trainingTime: trainingTime,
        epochs: this.config.TRANSFER_LEARNING.EPOCHS,
        finalLoss: history.history.loss[history.history.loss.length - 1],
        finalAccuracy: history.history.acc[history.history.acc.length - 1],
        bestLoss: this.trainingStats.bestLoss,
        bestAccuracy: this.trainingStats.bestAccuracy,
        modelSaved: true
      };
      
      return results;
      
    } catch (error) {
      console.error('❌ Error en entrenamiento:', error);
      throw new Error(`Training failed: ${error.message}`);
      
    } finally {
      this.isTraining = false;
    }
  }

  /**
   * Evalúa el modelo entrenado con dataset de prueba
   * @param {string} testDatasetPath - Ruta al dataset de prueba
   * @returns {Promise<Object>} Métricas de evaluación
   */
  async evaluateModel(testDatasetPath) {
    try {
      console.log('🧪 Evaluando modelo entrenado...');
      
      if (!this.plateDetector.isReady) {
        await this.plateDetector.initialize();
      }
      
      const { trainData: testData } = await this.prepareDataset(testDatasetPath);
      
      const evaluation = await this.plateDetector.model.evaluateDataset(testData);
      const [loss, accuracy] = await Promise.all([
        evaluation[0].data(),
        evaluation[1].data()
      ]);
      
      console.log(`📊 Evaluación: Loss = ${loss[0]}, Accuracy = ${accuracy[0]}`);
      
      return {
        loss: loss[0],
        accuracy: accuracy[0],
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      console.error('❌ Error en evaluación:', error);
      throw new Error(`Evaluation failed: ${error.message}`);
    }
  }

  /**
   * Obtiene estadísticas del entrenamiento actual
   * @returns {Object} Estadísticas de entrenamiento
   */
  getTrainingStats() {
    return {
      isTraining: this.isTraining,
      ...this.trainingStats,
      config: {
        epochs: this.config.TRANSFER_LEARNING.EPOCHS,
        batchSize: this.config.TRANSFER_LEARNING.BATCH_SIZE,
        learningRate: this.config.TRANSFER_LEARNING.LEARNING_RATE,
        validationSplit: this.config.TRANSFER_LEARNING.VALIDATION_SPLIT
      }
    };
  }

  // Métodos privados

  async _validateDatasetStructure(withPlatesPath, withoutPlatesPath) {
    try {
      await fs.access(withPlatesPath);
      await fs.access(withoutPlatesPath);
      
      const withPlatesFiles = await fs.readdir(withPlatesPath);
      const withoutPlatesFiles = await fs.readdir(withoutPlatesPath);
      
      if (withPlatesFiles.length === 0 || withoutPlatesFiles.length === 0) {
        throw new Error('Dataset directories cannot be empty');
      }
      
      console.log(`📁 Dataset válido: ${withPlatesFiles.length} con placas, ${withoutPlatesFiles.length} sin placas`);
      
    } catch (error) {
      throw new Error(`Invalid dataset structure: ${error.message}`);
    }
  }

  async _loadImagesFromDirectory(dirPath, label) {
    const files = await fs.readdir(dirPath);
    const imageFiles = files.filter(file => 
      /\.(jpg|jpeg|png|bmp)$/i.test(file)
    );
    
    const images = [];
    
    for (const file of imageFiles) {
      try {
        const imagePath = path.join(dirPath, file);
        const imageBuffer = await fs.readFile(imagePath);
        
        // Procesar imagen con Sharp y convertir a tensor
        const processedImage = await sharp(imageBuffer)
          .resize(224, 224)
          .removeAlpha()
          .raw()
          .toBuffer();
        
        // Convertir a tensor y normalizar
        const tensor = tf.tensor3d(
          new Uint8Array(processedImage), 
          [224, 224, 3], 
          'int32'
        ).div(255.0);
        
        images.push({ image: tensor, label });
        
      } catch (error) {
        console.warn(`⚠️ Error procesando ${file}:`, error.message);
      }
    }
    
    console.log(`📸 Cargadas ${images.length} imágenes de ${dirPath}`);
    return images;
  }

  _shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  _createTrainingCallbacks() {
    return {
      onEpochEnd: (epoch, logs) => {
        this.trainingStats.epochCount = epoch + 1;
        
        if (logs.loss < this.trainingStats.bestLoss) {
          this.trainingStats.bestLoss = logs.loss;
        }
        
        if (logs.acc > this.trainingStats.bestAccuracy) {
          this.trainingStats.bestAccuracy = logs.acc;
        }
        
        console.log(`🕒 Epoch ${epoch + 1}/${this.config.TRANSFER_LEARNING.EPOCHS}:`);
        console.log(`   📉 Loss: ${logs.loss.toFixed(4)} | Val Loss: ${logs.val_loss.toFixed(4)}`);
        console.log(`   📈 Acc: ${logs.acc.toFixed(4)} | Val Acc: ${logs.val_acc.toFixed(4)}`);
      }
    };
  }

  async _saveTrainedModel() {
    try {
      const modelPath = this.config.PATHS.CUSTOM_MODEL_PATH;
      
      // Crear directorio si no existe
      await fs.mkdir(path.dirname(modelPath), { recursive: true });
      
      // Guardar modelo
      await this.plateDetector.model.save(`file://${modelPath}`);
      
      console.log(`💾 Modelo guardado en: ${modelPath}`);
      
    } catch (error) {
      console.error('❌ Error guardando modelo:', error);
      throw new Error(`Model saving failed: ${error.message}`);
    }
  }

  /**
   * Libera recursos del servicio
   */
  async dispose() {
    try {
      await this.plateDetector.dispose();
      this.isTraining = false;
      console.log('🗑️ TrainingService liberado correctamente');
      
    } catch (error) {
      console.error('❌ Error liberando TrainingService:', error);
    }
  }
}

export default TrainingService;
