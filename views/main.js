// Elementos del DOM
const form = document.getElementById("formulario");
const resultado = document.getElementById("resultado");
const imagenSubida = document.getElementById("imagenSubida");
const videoElement = document.getElementById("videoElement");
const canvasElement = document.getElementById("canvasElement");
const startCameraButton = document.getElementById("startCamera");
const stopCameraButton = document.getElementById("stopCamera");
const startStreamButton = document.getElementById("startStream");
const stopStreamButton = document.getElementById("stopStream");

// Variables globales
let stream = null;
let streamInterval = null;
let statsInterval = null;
const ANALYSIS_INTERVAL = 500; // Analizar cada 0.5 segundos (2 FPS)
const ctx = canvasElement.getContext("2d");

// Estadísticas
let totalAnalyzed = 0;
let totalPlatesFound = 0;
let startTime = null;

// Función para iniciar la cámara
startCameraButton.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1280 }, // Mayor resolución para placas distantes
        height: { ideal: 720 }, // Mayor resolución para placas distantes
        frameRate: { max: 30 }, // Mantener buen frame rate
      },
      audio: false,
    });

    videoElement.srcObject = stream;

    // Ajustar tamaño del canvas al del video
    videoElement.onloadedmetadata = () => {
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
    };

    // Actualizar botones
    startCameraButton.disabled = true;
    stopCameraButton.disabled = false;
    startStreamButton.disabled = false;

    resultado.innerText += `
    ${new Date().toLocaleTimeString()} Cámara activada. Pulse 'Iniciar Análisis' para comenzar a detectar placas.
    `;
  } catch (error) {
    console.error("Error al acceder a la cámara:", error);
    resultado.innerText += `${new Date().toLocaleTimeString()} Error al acceder a la cámara: ${error.message}
    `;
  }
});

// Detener la cámara
stopCameraButton.addEventListener("click", () => {
  stopCamera();
  stopStreamAnalysis();
});

// Iniciar análisis en tiempo real
startStreamButton.addEventListener("click", () => {
  if (!stream) {
    resultado.innerText += `${new Date().toLocaleTimeString()} Por favor, active la cámara primero.
    `;
    return;
  }

  startStreamAnalysis();
});

// Detener análisis en tiempo real
stopStreamButton.addEventListener("click", () => {
  stopStreamAnalysis();
});

// Función para capturar frame y enviar al servidor
async function captureAndAnalyze() {
  if (!stream || !videoElement.videoWidth) return;

  // Dibujar el frame actual en el canvas
  ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

  // Convertir a blob
  try {
    const blob = await new Promise((resolve) => {
      canvasElement.toBlob(resolve, "image/jpeg", 0.95); // Aumentar calidad para mejor reconocimiento de placas distantes
    });

    if (!blob) {
      console.error("Error al crear blob de la imagen");
      return;
    }

    // Crear FormData y enviar al servidor
    const formData = new FormData();
    formData.append("image", blob, "realtime-capture.jpg");

    // Enviar sin mostrar estado para velocidad máxima
    const response = await fetch("/api/image/upload", {
      method: "POST",
      headers: {
        "X-Request-Source": "realtime",
      },
      body: formData,
    });

    const data = await response.json();

    // Actualizar estadísticas
    totalAnalyzed++;
    if (data.arrayReply && Array.isArray(data.arrayReply) && data.arrayReply.length > 0) {
      totalPlatesFound += data.arrayReply.length;
      resultado.innerText += `${new Date().toLocaleTimeString()} 🚗 PLACAS DETECTADAS: ${data.arrayReply.join(", ")}\n`;
    }
  } catch (error) {
    console.error("Error al analizar imagen:", error);
    resultado.innerText += `${new Date().toLocaleTimeString()} Error al analizar: ${
      error.message
    }
    `;
  }
}

// Función para actualizar estadísticas en UI
function updateStats() {
  const statsContent = document.getElementById('statsContent');
  if (startTime) {
    const elapsed = (Date.now() - startTime) / 1000;
    const fps = totalAnalyzed / elapsed;
    const successRate = totalAnalyzed > 0 ? (totalPlatesFound / totalAnalyzed * 100).toFixed(1) : 0;
    
    statsContent.innerHTML = `
      ⏱️ Tiempo transcurrido: ${elapsed.toFixed(0)}s<br>
      📊 Imágenes analizadas: ${totalAnalyzed}<br>
      🚗 Placas encontradas: ${totalPlatesFound}<br>
      ⚡ Velocidad: ${fps.toFixed(1)} FPS<br>
      ✅ Tasa de éxito: ${successRate}%
    `;
  }
}

// Iniciar análisis en streaming
function startStreamAnalysis() {
  if (streamInterval) clearInterval(streamInterval);
  if (statsInterval) clearInterval(statsInterval);

  // Reset estadísticas
  totalAnalyzed = 0;
  totalPlatesFound = 0;
  startTime = Date.now();

  // Programar captura periódica
  streamInterval = setInterval(captureAndAnalyze, ANALYSIS_INTERVAL);
  
  // Actualizar estadísticas cada segundo
  statsInterval = setInterval(updateStats, 1000);

  // Actualizar botones
  startStreamButton.disabled = true;
  stopStreamButton.disabled = false;

  resultado.innerText += `⚡ Análisis en tiempo real iniciado a 2 FPS...\n`;
}

// Detener análisis en streaming
function stopStreamAnalysis() {
  if (streamInterval) {
    clearInterval(streamInterval);
    streamInterval = null;
  }
  
  if (statsInterval) {
    clearInterval(statsInterval);
    statsInterval = null;
  }

  // Actualizar botones
  if (stream) {
    startStreamButton.disabled = false;
  }
  stopStreamButton.disabled = true;

  resultado.innerText += `Análisis en tiempo real detenido.\n`;
}

// Detener la cámara
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    videoElement.srcObject = null;
    stream = null;

    // Resetear botones
    startCameraButton.disabled = false;
    stopCameraButton.disabled = true;
    startStreamButton.disabled = true;
    stopStreamButton.disabled = true;

    resultado.innerText += `Cámara desactivada.
    `;
  }
}

// Subida manual de imagen
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = document.getElementById("imageInput").files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("image", file);

  resultado.innerText += `Analizando imagen subida...
  `;

  try {
    const res = await fetch("/api/image/upload", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    resultado.innerText +=
      data.arrayReply &&
      Array.isArray(data.arrayReply) &&
      data.arrayReply.length
        ? `Placas detectadas: ${data.arrayReply.join(", ")}
        `
        : `No se detectaron placas en la imagen.
        `;
  } catch (error) {
    console.error("Error al procesar imagen:", error);
    resultado.innerText += `Error al analizar: ${error.message}`;
  }
});

// Vista previa de la imagen subida
function previewImage(e) {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();

    reader.onload = (event) => {
      imagenSubida.src = event.target.result;
      imagenSubida.style.display = "block";
    };
    reader.readAsDataURL(file);
  } else {
    imagenSubida.style.display = "none";
    imagenSubida.src = "";
  }
}
