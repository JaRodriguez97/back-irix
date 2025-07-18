import { getGeminiReply } from "../services/geminiService.js";
import sharp from "sharp";

/**
 * Controlador para detectar placas en imágenes y transmisiones de video
 */
export const detectPlate = async (req, res) => {
  // Verificar si hay un archivo cargado
  if (!req.file || !req.file.buffer) {
    return res
      .status(400)
      .send({ error: "No se ha proporcionado una imagen válida" });
  }

  try {
    // Optimización ultra-rápida con sharp
    const buffer = await sharp(req.file.buffer)
      .resize({
        width: 1024,
        height: 576,
        fit: 'inside',
        withoutEnlargement: true
      })
      .jpeg({
        quality: 90, // Calidad alta para mejor precisión
        progressive: false, // Desactivar para velocidad
        optimizeScans: false, // Desactivar para velocidad
        mozjpeg: false // Desactivar para velocidad
      })
      .toBuffer();

    // Obtener la fuente de la solicitud (para propósitos de registro)
    const source = req.headers["x-request-source"] || "unknown";
    const isRealtime = source === "realtime";

    // Configurar opciones para Gemini según la fuente
    const options = {
      outputFormat: "array",
      prompt: isRealtime
        ? `Analiza esta imagen en tiempo real y detecta las posibles placas de vehiculos de Colombia, que son de tipo ABC123, donde ABC son letras y 123 son números. Devuelve solo las placas encontradas en formato ABC123 en un array con el mismo formato. Ignora cualquier placa que no tenga este formato antes mensionado (ABC123). No incluyas ningún otro texto o explicación. Si no encuentras ninguna placa, responde con un @.`
        : undefined, // Usar el prompt predeterminado para subidas normales
    };

    // Enviar a Gemini para análisis
    let reply = await getGeminiReply(buffer, options);

    // Procesar la respuesta
    reply = reply
      .replace(/\n/g, "")
      .replace(/```json\s*/, "") // elimina ```json y posibles espacios/líneas
      .replace(/```/, "") // elimina la marca de cierre ```
      .trim();

    console.log(`🔍 Respuesta de Gemini (${source}):`, reply);

    // Intentar parsear la respuesta como JSON
    let arrayReply = [];
    if (reply && reply !== "@") {
      try {
        arrayReply = JSON.parse(reply);
        if (!Array.isArray(arrayReply)) {
          arrayReply = [arrayReply];
        }
      } catch (parseError) {
        // Si no se puede parsear como JSON, intentar extraer placas mediante regex
        console.log("⚠️ Error al parsear JSON, intentando extraer con regex");
        const plateRegex = /[A-Z]{3}\d{3}/g;
        const matches = reply.match(plateRegex);
        if (matches) {
          arrayReply = matches;
        }
      }
    }

    // Enviar respuesta
    return res.status(200).json({ arrayReply });
  } catch (err) {
    console.error("❌ Error procesando imagen:", err);
    // Enviar un código de estado 200 para no interrumpir la interfaz de usuario
    // pero con un mensaje de error para informar al cliente
    return res.status(200).json({
      error: "Error al procesar la imagen",
      arrayReply: [],
    });
  }
};
