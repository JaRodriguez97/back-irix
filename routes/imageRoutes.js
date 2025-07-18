import { Router } from "express";
const router = Router();
import multer, { memoryStorage } from "multer";
import { detectPlate } from "../controllers/imageController.js";

// Configuración de multer (archivo temporal)
const storage = memoryStorage(); // no guarda en disco
const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB para permitir imágenes de mayor calidad
  fileFilter: (req, file, cb) => {
    if (!file.mimetype.startsWith("image/")) {
      return cb(new Error("Archivo inválido"));
    }
    cb(null, true);
  },
});

router.post("/upload", upload.single("image"), detectPlate);

export default router;
