import { Router } from "express";
const router = Router();

/**
 * DETECCIÓN DE PLACAS VÍA SOCKET.IO
 * 
 * ✅ Las imágenes se procesan en tiempo real vía Socket.IO
 * ❌ No se necesita multer ni rutas HTTP para imágenes
 * 
 * Flujo actual:
 * 1. Frontend captura imagen → Blob/ArrayBuffer
 * 2. SocketService.sendImageForAnalysis() → Envía vía socket
 * 3. ImageHandler.processImageFromSocket() → Procesa directamente
 * 4. DetectionService.detectPlate() → Retorna resultado
 */

// Endpoint de estado del servicio
router.get("/status", (req, res) => {
  res.json({ 
    service: "Detección de Placas Vehiculares",
    method: "Socket.IO Real-time",
    status: "active",
    timestamp: new Date().toISOString(),
    endpoints: {
      socketConnection: "ws://localhost:3000",
      events: {
        send: "analyze-image",
        receive: "analysis-result"
      }
    }
  });
});

export default router;
