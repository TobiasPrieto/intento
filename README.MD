# Trabajo Práctico 2 - Inferencia de Clima en Australia

Este contenedor ejecuta el pipeline de inferencia para predecir si lloverá mañana en Australia.

## Contenido
- `inferencia.py`: Script principal que carga modelos y procesa datos.
- `Dockerfile`: Configuración de la imagen.
- `*.joblib`: Modelos y transformadores serializados.

## Instrucciones

### 1. Construir la imagen (Build)
Abre una terminal en esta carpeta (`docker`) y ejecuta:

```bash
docker build -t prediccion-clima .