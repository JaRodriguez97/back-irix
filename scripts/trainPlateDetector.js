#!/usr/bin/env node

/**
 * Script de Entrenamiento del Modelo de Detección de Placas
 * Ejemplo de uso del TrainingService para transfer learning
 */

import TrainingService from '../services/TrainingService.js';
import path from 'path';

async function main() {
  const trainingService = new TrainingService('development');
  
  try {
    console.log('🚀 Iniciando entrenamiento del detector de placas vehiculares');
    console.log('=' .repeat(60));
    
    // Ruta al dataset (debe existir con estructura: si/ y no/)
    const datasetPath = './datasets';
    
    console.log(`📂 Dataset: ${datasetPath}`);
    console.log(`📊 Configuración de entrenamiento:`);
    
    const stats = trainingService.getTrainingStats();
    console.log(`   • Epochs: ${stats.config.epochs}`);
    console.log(`   • Batch Size: ${stats.config.batchSize}`);
    console.log(`   • Learning Rate: ${stats.config.learningRate}`);
    console.log(`   • Validation Split: ${stats.config.validationSplit}`);
    console.log('');
    
    // Entrenar modelo
    const results = await trainingService.trainModel(datasetPath);
    
    console.log('');
    console.log('🎉 ENTRENAMIENTO COMPLETADO');
    console.log('=' .repeat(60));
    console.log(`✅ Éxito: ${results.success}`);
    console.log(`⏱️ Tiempo total: ${Math.round(results.trainingTime / 1000)}s`);
    console.log(`📈 Accuracy final: ${(results.finalAccuracy * 100).toFixed(2)}%`);
    console.log(`📉 Loss final: ${results.finalLoss.toFixed(4)}`);
    console.log(`🏆 Mejor accuracy: ${(results.bestAccuracy * 100).toFixed(2)}%`);
    console.log(`🎯 Mejor loss: ${results.bestLoss.toFixed(4)}`);
    console.log(`💾 Modelo guardado: ${results.modelSaved}`);
    console.log('');
    
    // Opcional: Evaluar con dataset de test
    const testDatasetPath = './datasets_test';
    console.log('🧪 Evaluando modelo con dataset de prueba...');
    
    try {
      const evaluation = await trainingService.evaluateModel(testDatasetPath);
      console.log(`📊 Evaluación en test set:`);
      console.log(`   • Accuracy: ${(evaluation.accuracy * 100).toFixed(2)}%`);
      console.log(`   • Loss: ${evaluation.loss.toFixed(4)}`);
      
    } catch (evalError) {
      console.log('⚠️ Dataset de prueba no disponible, saltando evaluación');
    }
    
    console.log('');
    console.log('✅ Proceso completado exitosamente');
    
  } catch (error) {
    console.error('');
    console.error('❌ ERROR EN ENTRENAMIENTO');
    console.error('=' .repeat(60));
    console.error('Mensaje:', error.message);
    console.error('');
    
    if (error.message.includes('Dataset preparation failed')) {
      console.error('💡 SOLUCIÓN SUGERIDA:');
      console.error('1. Verifica que el directorio del dataset existe');
      console.error('2. Estructura requerida:');
      console.error('   datasets/');
      console.error('   ├── si/    (imágenes CON placas: si(1).jpeg, si(2).jpeg...)');
      console.error('   └── no/    (imágenes SIN placas: no(1).jpeg, no(2).jpeg...)');
      console.error('3. Formatos soportados: jpg, jpeg, png, bmp');
      console.error('');
    }
    
    process.exit(1);
    
  } finally {
    // Limpiar recursos
    await trainingService.dispose();
  }
}

// Manejar señales de interrupción
process.on('SIGINT', async () => {
  console.log('\n⚠️ Entrenamiento interrumpido por el usuario');
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n⚠️ Entrenamiento terminado por el sistema');
  process.exit(0);
});

// Ejecutar script
main().catch(console.error);
