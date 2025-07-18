import "dotenv/config";
import { env } from "process";

// Configuración de API keys y versiones
const API_KEYS = [
  env.API_KEY,
  env.API_KEY2,
  env.API_KEY3,
  env.API_KEY4
];

const VERSIONS = ["2.0", "2.5"];

// Estado de rotación
let currentKeyIndex = 0;
let currentVersionIndex = 0;

// Contador de peticiones para estadísticas
let requestCount = 0;

// Pool de workers para procesamiento concurrente
const MAX_CONCURRENT_REQUESTS = 20; // Máximo de peticiones concurrentes
let activeRequests = 0;
const requestQueue = [];

// Métricas de rendimiento
let processingTimes = [];
let successfulRequests = 0;
let failedRequests = 0;

// Función para obtener estadísticas
export function getRotationStats() {
  const avgProcessingTime = processingTimes.length > 0 
    ? processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length 
    : 0;
  
  return {
    totalRequests: requestCount,
    successfulRequests,
    failedRequests,
    activeRequests,
    queuedRequests: requestQueue.length,
    avgProcessingTime: Math.round(avgProcessingTime),
    currentKeyIndex: currentKeyIndex + 1,
    currentVersionIndex: currentVersionIndex,
    currentVersion: VERSIONS[currentVersionIndex],
    totalKeys: API_KEYS.length,
    totalVersions: VERSIONS.length,
    maxRequestsPerMinute: API_KEYS.length * VERSIONS.length * 15, // 4 keys x 2 versions x 15 req/min
    maxConcurrentRequests: MAX_CONCURRENT_REQUESTS
  };
}

// Función para procesar cola de peticiones
function processQueue() {
  if (requestQueue.length === 0 || activeRequests >= MAX_CONCURRENT_REQUESTS) {
    return;
  }
  
  const { resolve, reject, buffer, options } = requestQueue.shift();
  activeRequests++;
  
  executeGeminiRequest(buffer, options)
    .then(resolve)
    .catch(reject)
    .finally(() => {
      activeRequests--;
      processQueue(); // Procesar siguiente en cola
    });
}

// Función interna para ejecutar petición a Gemini
async function executeGeminiRequest(buffer, options = {}) {
  const startTime = Date.now();
  
  try {
    const result = await getGeminiReplyInternal(buffer, options);
    
    // Métricas
    const processingTime = Date.now() - startTime;
    processingTimes.push(processingTime);
    if (processingTimes.length > 100) processingTimes.shift(); // Mantener solo últimas 100
    successfulRequests++;
    
    return result;
  } catch (error) {
    failedRequests++;
    throw error;
  }
}

// Función para obtener el próximo endpoint
function getNextEndpoint() {
  const currentKey = API_KEYS[currentKeyIndex];
  const currentVersion = VERSIONS[currentVersionIndex];
  
  const endpoint = `${env.URI_BASE}${currentVersion}-flash:generateContent?key=${currentKey}`;
  
  console.log(`🔑 Usando API Key ${currentKeyIndex + 1} con versión ${currentVersion}`);
  
  // Rotar versión
  currentVersionIndex = (currentVersionIndex + 1) % VERSIONS.length;
  
  // Si completamos el ciclo de versiones, rotar API key
  if (currentVersionIndex === 0) {
    currentKeyIndex = (currentKeyIndex + 1) % API_KEYS.length;
  }
  
  return endpoint;
}

/**
 * Función pública para procesar imagen con cola concurrente
 * @param {Buffer} buffer - Buffer de la imagen a analizar
 * @param {Object} options - Opciones de configuración
 * @returns {Promise<string>}
 */
export const getGeminiReply = async (buffer, options = {}) => {
  requestCount++;
  
  // Si podemos procesar inmediatamente, hacerlo
  if (activeRequests < MAX_CONCURRENT_REQUESTS) {
    activeRequests++;
    
    try {
      const result = await executeGeminiRequest(buffer, options);
      return result;
    } finally {
      activeRequests--;
      processQueue(); // Procesar siguiente en cola
    }
  }
  
  // Si no, agregar a la cola
  return new Promise((resolve, reject) => {
    requestQueue.push({ resolve, reject, buffer, options });
    console.log(`⏳ Petición agregada a cola. Cola: ${requestQueue.length}, Activas: ${activeRequests}`);
  });
};

/**
 * Función interna que ejecuta la petición real a Gemini
 * @param {Buffer} buffer - Buffer de la imagen a analizar
 * @param {Object} options - Opciones de configuración
 * @returns {Promise<string>}
 */
async function getGeminiReplyInternal(buffer, options = {}) {
  const base64Image = buffer.toString("base64");
  const { outputFormat = "array", prompt } = options;

  // Prompt predeterminado para detectar placas colombianas
  const defaultPrompt =
    "necesito que valides en toda la imagen, las posibles placas de vehiculos de colombia, que son de tipo ABC123, donde ABC son letras y 123 son números. Devuelve solo las placas encontradas en formato ABC123 en un array. No incluyas ningún otro texto o explicación. Si no encuentras ninguna placa, responde con '@'";

  const contents = [
    {
      role: "user",
      parts: [
        {
          inlineData: {
            mimeType: "image/jpeg",
            data: base64Image,
          },
        },
        {
          text: prompt || defaultPrompt,
        },
      ],
    },
  ];

  // Obtener el próximo endpoint con rotación
  const endpoint = getNextEndpoint();
  
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.1, // Mínimo para máxima velocidad y consistencia
          topK: 16, // Reducido para velocidad
          topP: 0.8, // Reducido para velocidad
          maxOutputTokens: 64, // Mínimo necesario para placas
        },
      }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(
      `Error en respuesta de Gemini: ${response.status}`,
      errorText
    );
    throw new Error(`Error en la API de Gemini: ${response.status}`);
  }

  const data = await response.json();
  console.log("🚀 ~ getGeminiReply ~ data:", data);

  const reply = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
  return reply;
}
