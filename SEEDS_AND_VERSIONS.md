# Información de Reproducibilidad

## Semillas Utilizadas

Para garantizar la reproducibilidad de los resultados, se han configurado las siguientes semillas:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

## Versiones de Dependencias

### Python
- **Versión requerida:** Python 3.8+
- **Versión recomendada:** Python 3.9+

### Dependencias Principales

```
pandas>=1.5.0
numpy>=1.21.0
transformers>=4.21.0
torch>=1.12.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Modelos de HuggingFace

- **Modelo principal:** `facebook/bart-large-mnli`
- **Versión del tokenizer:** Automática con transformers
- **Cache local:** Los modelos se descargan automáticamente en la primera ejecución

## Configuración del Entorno

### Sistema Operativo
- **Desarrollado en:** macOS
- **Compatible con:** Linux, Windows, macOS

### Hardware Recomendado
- **RAM:** 8GB mínimo (16GB recomendado)
- **Almacenamiento:** 5GB libres para modelos y datos
- **CPU:** Multicores recomendado para inferencia

### Configuraciones Específicas

#### Pandas
```python
# Configuración para muestreo reproducible
df_sample = df_full.groupby('Categoría').apply(
    lambda x: x.sample(n=min(167, len(x)), random_state=SEED)
).reset_index(drop=True)
```

#### Scikit-learn
```python
# Todas las métricas utilizan los mismos parámetros
accuracy_score(y_true, y_pred)
confusion_matrix(y_true, y_pred, labels=['Positivo', 'Negativo', 'Neutral'])
```

#### Matplotlib/Seaborn
```python
# Configuración de figuras
plt.figure(figsize=(12, 5))
plt.savefig('out/confusion_matrices.png', dpi=300, bbox_inches='tight')
```

## Notas de Reproducibilidad

1. **Internet requerido:** Primera ejecución requiere descarga del modelo BART
2. **Tiempo de ejecución:** 5-10 minutos para clasificar 500 oraciones
3. **Variabilidad:** Los resultados pueden variar ligeramente debido a actualizaciones del modelo
4. **Cache:** Los modelos se almacenan en cache local para ejecuciones posteriores

## Validación de Resultados

### Métricas Esperadas
- **Accuracy:** Entre 0.70 y 0.85 típicamente
- **Tiempo de procesamiento:** ~1-2 segundos por oración
- **Guardrails activados:** 5-15 casos en promedio

### Archivos Generados
- `out/confusion_matrices.png`
- `out/confidence_analysis.png`
- `out/metricas_resumen.csv`

## Troubleshooting

### Problemas Comunes

1. **Error de memoria:** Reducir batch size o usar CPU en lugar de GPU
2. **Modelo no descarga:** Verificar conexión a internet y permisos
3. **Resultados diferentes:** Verificar que todas las semillas estén configuradas

### Comandos de Verificación

```bash
# Verificar instalación
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print(torch.__version__)"

# Verificar semillas
python -c "import random, numpy as np; random.seed(42); np.random.seed(42); print('Seeds OK')"
```

## Fecha de Última Actualización

**Fecha:** Septiembre 2025  
**Versión del proyecto:** 1.0  
**Estado:** Estable

## Entorno de Ejecución

### Sistema Operativo
- **OS:** macOS
- **Fecha de ejecución:** Septiembre 20, 2025
- **Python:** 3.13 (recomendado 3.8+)

### Versiones Específicas Instaladas

Durante la ejecución del proyecto se instalaron las siguientes versiones:

```
transformers==4.56.2
huggingface-hub==0.35.0
tokenizers==0.22.1
safetensors==0.6.2
torch (compatible con Apple Silicon MPS)
pandas>=2.3.0
numpy>=2.3.1
matplotlib>=3.6.0
seaborn>=0.13.2
scikit-learn>=1.2.0
```

## Configuración de Reproducibilidad

### Semillas de Aleatoriedad

Todas las operaciones aleatorias están controladas con:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

### Determinismo

- **Muestreo de datos:** Seed fija para `df.sample(random_state=SEED)`
- **Orden de procesamiento:** Consistente con índices fijos
- **Modelo de HuggingFace:** Determinista en inferencia con mismos inputs

## Hardware Utilizado

- **Aceleración:** Apple Silicon MPS (Metal Performance Shaders)
- **Memoria:** Suficiente para procesar 500 oraciones con BART-large
- **Tiempo de ejecución:** ~1.5 minutos para clasificación completa

## Notas de Reproducibilidad

1. **Modelo base:** `facebook/bart-large-mnli` es determinista
2. **Orden de datos:** Mantenido con semillas fijas
3. **Versiones de bibliotecas:** Especificadas en requirements.txt
4. **Configuración GPU:** Automática detección de MPS en Apple Silicon

**Para reproducir exactamente estos resultados:**
1. Usar Python 3.8+
2. Instalar dependencias exactas: `pip install -r requirements.txt`
3. Ejecutar notebook completo en orden
4. Verificar que SEED = 42 esté configurado
