# Proyecto 1: Clasificación Zero-Shot con Guardrails

**Curso:** CC0C2 - Procesamiento de Lenguaje Natural  
**Estudiante:** Carlos Daniel Malvaceda Canales  
**Institución:** Universidad Nacional de Ingeniería  
**Fecha:** Septiembre 2025

## Descripción

Este proyecto implementa un sistema de clasificación zero-shot para análisis de sentimientos en textos en español utilizando modelos fundacionales de HuggingFace. El sistema incluye guardrails para mejorar la robustez de las predicciones.

## Objetivos

- Implementar clasificación zero-shot usando BART-large-MNLI
- Comparar el rendimiento de diferentes formulaciones de prompts
- Desarrollar guardrails para detectar y manejar casos problemáticos
- Evaluar el sistema con métricas estándar de clasificación
- Analizar errores y proponer mejoras

## Estructura del Proyecto

```
PC1-ZeroShot-Guardrails/
├── data/
│   └── nlp_prueba_cc0c2_large.csv    # Dataset de 10,000 oraciones
├── out/
│   ├── confusion_matrices.png        # Matrices de confusión
│   ├── confidence_analysis.png       # Análisis de confianza
│   └── metricas_resumen.csv          # Resumen de métricas
├── notebook.ipynb                    # Implementación principal
├── generate_dataset.py               # Script para generar datos
├── requirements.txt                  # Dependencias
└── README.md                         # Documentación
```

## Historias de Usuario

### Historia 1: Análisis de Sentimientos Automático
**Como** estudiante de ciencias de la computación procesando comentarios de redes sociales,  
**Quiero** clasificar automáticamente el sentimiento de textos en español sin entrenar un modelo específico,  
**Para** analizar rápidamente grandes volúmenes de opiniones y feedback.

### Historia 2: Validación de Resultados con Guardrails
**Como** desarrollador de sistemas de NLP,  
**Quiero** implementar guardrails que detecten casos problemáticos en la clasificación,  
**Para** mejorar la confiabilidad y robustez del sistema en producción.

### Historia 3: Comparación de Estrategias de Prompting
**Como** investigador en NLP,  
**Quiero** comparar diferentes formulaciones de prompts para zero-shot classification,  
**Para** optimizar el rendimiento del modelo sin datos de entrenamiento adicionales.

## Definition of Done (DoD)

✅ **Implementación técnica completada:**
- [ ] Sistema de clasificación zero-shot funcional con HuggingFace
- [ ] Evaluación de exactamente 500 oraciones del dataset
- [ ] Implementación de 2 prompts distintos ("sentimiento" vs "emoción")
- [ ] Guardrail funcional para detección de nombres propios

✅ **Métricas y evaluación:**
- [ ] Cálculo de accuracy para ambos prompts
- [ ] Matrices de confusión generadas y guardadas
- [ ] Identificación y análisis de 5 ejemplos de errores
- [ ] Comparación cuantitativa entre prompts

✅ **Reproducibilidad y documentación:**
- [ ] Semillas fijas documentadas en SEEDS_AND_VERSIONS.md
- [ ] Código ejecutable sin errores
- [ ] Resultados guardados en directorio `out/`
- [ ] README con instrucciones claras de ejecución

✅ **Entregables del curso:**
- [ ] Notebook con código comentado y preguntas teóricas respondidas
- [ ] Video de 5-10 minutos en formato sprint
- [ ] Mínimo 5 commits con mensajes en español
- [ ] Repositorio Git público en GitHub

## Instalación y Ejecución

### Prerrequisitos

- Python 3.8+
- pip o conda
- Conexión a internet (para descarga del modelo BART)

### Tiempos Estimados

- **Configuración inicial:** 5 minutos
- **Ejecución completa:** 10-15 minutos
- **Clasificación de 500 oraciones:** 5-10 minutos
- **Generación de gráficos:** 2-3 minutos

### Configuración del Entorno

1. Clonar el repositorio:
```bash
git clone https://github.com/danyelliot/PC1-ZeroShot-Guardrails.git
cd PC1-ZeroShot-Guardrails
```

2. Crear entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

### Generación del Dataset

Si necesitas regenerar el dataset:

```bash
python generate_dataset.py
```

### Ejecución del Análisis

1. Abrir Jupyter Notebook:
```bash
jupyter notebook notebook.ipynb
```

2. Ejecutar todas las celdas en orden

**Nota:** La clasificación de 500 oraciones puede tomar 5-10 minutos debido al procesamiento con el modelo BART.

## Metodología

### Modelo Utilizado

- **Modelo:** facebook/bart-large-mnli
- **Tipo:** Zero-shot classification
- **Idioma objetivo:** Español
- **Categorías:** Positivo, Negativo, Neutral

### Prompts Evaluados

1. **Prompt 1:** ["sentimiento positivo", "sentimiento negativo", "sentimiento neutral"]
2. **Prompt 2:** ["emoción positiva", "emoción negativa", "emoción neutral"]

### Guardrails Implementados

**Objetivo:** Mejorar la robustez del sistema detectando casos problemáticos

**Implementación:**
- **Detector de nombres propios:** Regex `r'\b(?<!^)(?<!\. )[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b'`
- **Lógica de ajuste:** Si se detectan nombres propios y la confianza es baja (<0.6), la predicción se reclasifica como "Neutral"
- **Rationale:** Los nombres propios pueden generar clasificaciones incorrectas por asociaciones no relacionadas con el sentimiento

**Casos que maneja:**
- Textos con nombres de personas, lugares o marcas
- Situaciones donde el modelo tiene baja confianza
- Prevención de sesgos por entidades específicas

## Resultados Principales

### Implementación (3 puntos)
✅ **HuggingFace pipeline:** Utiliza `facebook/bart-large-mnli` para zero-shot classification  
✅ **500 oraciones:** Muestreo estratificado del dataset `nlp_prueba_cc0c2_large.csv`  
✅ **2 prompts distintos:** "Clasifica el sentimiento" vs "Evalúa la emoción"  
✅ **Guardrail implementado:** Detección de nombres propios con regex  
✅ **Métricas calculadas:** Accuracy y matriz de confusión para ambos prompts  

### Teoría (respuestas escritas en el notebook)
1. **Modelo fundacional y pretraining:** Definición y conceptos clave
2. **In-context learning en zero-shot:** Explicación del mecanismo
3. **Riesgos de prompt injection:** Identificación y mitigación
4. **Impacto de tokens en costo computacional:** Análisis cuantitativo
5. **Análisis de fallo de clasificación:** Caso específico con solución propuesta

### Entregables
✅ **Notebook completo:** `notebook.ipynb` con código, métricas y teoría  
✅ **Video sprint:** Demostración de pipeline, resultados y guardrail  
✅ **Repositorio Git:** 7 commits con mensajes descriptivos en español  
✅ **Datos y resultados:** Dataset en `data/` y outputs en `out/`  

## Resultados Principales

- **Accuracy máximo:** Variable según la ejecución (típicamente 70-80%)
- **Mejor formulación:** Se determina empíricamente comparando ambos prompts
- **Guardrails:** Efectivos para casos con nombres propios
- **Análisis de errores:** Identificación de 5 casos problemáticos con explicaciones
- **Tiempo de procesamiento:** Aproximadamente 5-10 minutos para 500 oraciones

## Preguntas Teóricas (Respondidas en el Notebook)

1. **¿Qué es un modelo fundacional y cómo se relaciona con el pretraining?**
   - Análisis conceptual de modelos base y su entrenamiento inicial

2. **¿Cómo funciona el in-context learning en clasificación zero-shot?**
   - Explicación del mecanismo de aprendizaje sin ejemplos específicos

3. **¿Cuáles son los principales riesgos de prompt injection en este contexto?**
   - Identificación de vulnerabilidades y estrategias de mitigación

4. **¿Cómo impacta el número de tokens en el costo computacional?**
   - Análisis cuantitativo de complejidad y recursos

5. **Analiza un caso específico de fallo en la clasificación y propón una solución**
   - Estudio de caso con error real y mejora propuesta

## Estructura del Código

### Componentes Principales

1. **Configuración reproducible:** Semillas fijas para reproducibilidad
2. **Carga de datos:** Muestreo estratificado de 500 oraciones
3. **Pipeline zero-shot:** Configuración con transformers de HuggingFace
4. **Guardrails:** Implementación de reglas de validación
5. **Evaluación:** Métricas estándar y análisis de errores
6. **Visualización:** Matrices de confusión y análisis de confianza

### Funciones Clave

- `detect_proper_nouns()`: Detecta nombres propios usando regex
- `apply_guardrail()`: Aplica lógica de guardrails
- `classify_with_prompts()`: Realiza clasificación con prompts específicos

## Archivos de Salida

- `out/confusion_matrices.png`: Matrices de confusión para ambos prompts
- `out/confidence_analysis.png`: Distribución de confianzas y accuracy por rangos
- `out/metricas_resumen.csv`: Resumen cuantitativo de resultados

## Dependencias Principales

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

## Reproducibilidad

El proyecto está configurado para ser completamente reproducible:

- Semilla fija (`SEED = 42`) para todos los componentes aleatorios
- Versiones específicas de dependencias en `requirements.txt`
- Documentación detallada de la metodología

## Limitaciones

- El modelo BART-MNLI fue entrenado principalmente en inglés
- Evaluación limitada a 500 oraciones
- Guardrails básicos (solo nombres propios)
- Sin fine-tuning específico para el dominio

## Trabajo Futuro

1. Fine-tuning del modelo con datos específicos del dominio
2. Implementación de guardrails más sofisticados
3. Evaluación con datasets más grandes
4. Optimización de prompts mediante búsqueda sistemática
5. Ensemble de múltiples modelos

## Contacto

**Carlos Daniel Malvaceda Canales**  
Universidad Nacional de Ingeniería  
Curso: CC0C2 - Procesamiento de Lenguaje Natural

## Licencia

Este proyecto es parte de una práctica calificada académica. Todos los derechos reservados.
