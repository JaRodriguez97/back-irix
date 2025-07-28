/**
 * Configuración y manejo del servidor Socket.IO
 */

import { Server } from 'socket.io';
import {
    SOCKET_CONFIG,
    CLIENT_EVENTS,
    SERVER_EVENTS,
    SERVER_STATUS,
    ERROR_CODES,
    IMAGE_CONFIG
} from './SocketEvents.js';
import { imageHandler } from './ImageHandler.js';

export const initializeSocketServer = async (httpServer) => {
    const io = new Server(httpServer, {
        cors: SOCKET_CONFIG.CORS,
        ...SOCKET_CONFIG.CONNECTION
    });

    // Inicializar ImageHandler
    try {
        await imageHandler.initialize();
        console.log('🔌 Socket.IO configurado con ImageHandler');
    } catch (error) {
        console.error('❌ Error inicializando ImageHandler:', error);
        throw error;
    }

    // Almacena el estado del servidor
    let serverStatus = SERVER_STATUS.READY;

    io.on(CLIENT_EVENTS.CONNECT, (socket) => {
        console.log('Nuevo cliente conectado: ', socket.id);

        socket.emit(SERVER_EVENTS.CONNECTION_CONFIRMED);

        // Manejo de eventos de análisis de imagen
        socket.on(CLIENT_EVENTS.ANALYZE_IMAGE, (imageData) => {
            if (serverStatus !== SERVER_STATUS.READY) {
                return socket.emit(SERVER_EVENTS.ANALYSIS_ERROR, {
                    code: ERROR_CODES.INTERNAL_ERROR,
                    message: 'Servidor no está listo para procesar.'
                });
            }
            
            // Cambiar estado a procesando
            serverStatus = SERVER_STATUS.PROCESSING;
            console.log(`Procesando imagen de ${socket.id}`);

// Procesamiento real usando ImageHandler
            imageHandler.processImageFromSocket(imageData, socket.id)
                .then(result => {
                    socket.emit(SERVER_EVENTS.ANALYSIS_RESULT, result.result);
                    serverStatus = SERVER_STATUS.READY;
                })
                .catch(error => {
                    socket.emit(SERVER_EVENTS.ANALYSIS_ERROR, {
                        code: error.code || ERROR_CODES.INTERNAL_ERROR,
                        message: error.message
                    });
                    serverStatus = SERVER_STATUS.READY;
                });
        });

        socket.on(CLIENT_EVENTS.DISCONNECT, () => {
            console.log('Cliente desconectado: ', socket.id);
        });
    });

    // Publicar estado del servidor cada 10 segundos
    setInterval(() => {
        io.emit(SERVER_EVENTS.SERVER_STATUS, {
            status: serverStatus
        });
    }, 10000);
};
