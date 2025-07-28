# PLAN DE REFACTORING - IRIX BACKEND

## ELEMENTOS A ELIMINAR
- [x] Archivo `eng.traineddata` (OCR innecesario para Fase 1) - NO ENCONTRADO
- [x] Archivo `spa.traineddata` (OCR innecesario para Fase 1) - NO ENCONTRADO
- [x] Servicio `huggingfaceService.js` (redundante con TensorFlow) - NO ENCONTRADO
- [x] Servicio `llamaService.js` (redundante con Gemini) - ELIMINADO
- [x] Implementaciones sin usar en `/implementations/` - TODAS NECESARIAS
- [ ] Dependencias innecesarias en package.json

## ELEMENTOS A REUTILIZAR Y OPTIMIZAR
- [x] `plateDetectionService.js` - OPTIMIZAR para 2fps
- [x] `geminiService.js` - MANTENER rotación de API keys
- [x] `websocketService.js` - ELIMINADO (reemplazado por Socket.IO)
- [x] `imageController.js` - SEPARAR fases 1 y 2
- [x] Interfaces en `/interfaces/` - APLICAR principios SOLID
- [x] Estadísticas en `/implementations/PlateStatisticsManager.js`

## ELEMENTOS A AJUSTAR
- [x] `app.js` - Configurar para producción 2fps
- [x] Rate limiting para 120 req/min (2fps)
- [x] Buffer de memoria optimizado para imágenes HD
- [x] Configuraciones Sharp para máximo rendimiento
- [x] TensorFlow.js con modelo específico para placas

## ARCHIVOS A CREAR
- [x] `/models/PlateDetectionModel.js` - Single Responsibility
- [x] `/factories/ModelFactory.js` - Factory Pattern
- [x] `/strategies/DetectionStrategy.js` - Strategy Pattern
- [ ] `/observers/PerformanceObserver.js` - Observer Pattern
- [ ] `/adapters/GeminiAdapter.js` - Adapter Pattern
- [ ] `/services/ImageProcessingService.js` - Dependency Inversion
- [ ] `/validators/ImageValidator.js` - Open/Closed Principle
- [x] `/config/DetectionConfig.js` - Configuration Management

## PRINCIPIOS SOLID APLICADOS
1. **Single Responsibility**: Cada clase tiene una sola razón para cambiar
2. **Open/Closed**: Extensible sin modificar código existente
3. **Liskov Substitution**: Implementaciones intercambiables
4. **Interface Segregation**: Interfaces específicas por funcionalidad
5. **Dependency Inversion**: Depende de abstracciones, no concreciones

## OPTIMIZACIONES TIEMPO REAL (2FPS)
- Modelo TensorFlow.js específico para placas vehiculares
- Buffer pool para reutilizar memoria
- Worker threads para procesamiento paralelo
- Cache de resultados para imágenes similares
- Compresión adaptativa según ancho de banda
- Predicción de movimiento vehicular
