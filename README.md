# Proyecto-Final-Computer-Vision-I
## Sistema de Visión por Computador para Seguridad y Seguimiento Interactivo

Este repositorio contiene el desarrollo de un sistema completo de visión por computador realizado como proyecto final de la asignatura. El sistema integra técnicas clásicas de procesamiento de imagen para implementar un mecanismo de seguridad visual y un sistema de seguimiento interactivo en tiempo real aplicado a un juego de *Piedra, Papel o Tijera*.

El proyecto ha sido desarrollado utilizando *Python y OpenCV*, evitando el uso de modelos de aprendizaje profundo en el flujo principal, con el objetivo de reforzar la comprensión de los fundamentos clásicos de la visión por computador.

---

## Funcionalidades principales

El sistema se estructura en los siguientes módulos:

- *Calibración de cámara*
  - Obtención de parámetros intrínsecos y coeficientes de distorsión a partir de un tablero de ajedrez.
- *Sistema de seguridad*
  - Detección de patrones visuales (códigos QR).
  - Decodificación de una secuencia visual de desbloqueo.
- *Extracción de información*
  - Extracción de información simbólica (seguridad).
  - Extracción de características geométricas de la mano (tracking).
- *Sistema de seguimiento (Tracker)*
  - Segmentación de manos mediante sustracción de fondo.
  - Análisis de contornos, envolvente convexa y defectos de convexidad.
  - Clasificación de gestos para el juego de Piedra, Papel o Tijera.
  - Estabilización temporal mediante votación mayoritaria.
- *Salida de vídeo en tiempo real*
  - Visualización de resultados, marcador del juego y FPS suavizados.
- *Ampliaciones*
  - Implementación experimental alternativa usando MediaPipe para detección de manos.

---

## Estructura del repositorio

```text
├── calibracion.py          # Módulo de calibración de la cámara
├── desbloquear.py          # Sistema de seguridad basado en patrones visuales
├── trackerfinal.py              # Sistema de seguimiento y juego Piedra, Papel o Tijera
├── main.py                 # Archivo principal de ejecución
│
├── ImagenesCalibracion/    # Imágenes del tablero para calibración
│
├── tracker_ampliacion.py   # Ampliación del tracker usando Mediapipe   
├── requirements.txt        # Las librerias necesarias para la ejecución del proyecto
├── README.md               # Este archivo