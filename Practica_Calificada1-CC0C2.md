## Práctica calificada 1-CC0C2

### **Instrucciones generales**:
- **Plazo**: 10 días. Entrega un repositorio Git con todos los entregables.
- **Tiempo estimado**: Cada proyecto ~6 horas (3 h implementación, 1.5 h teoría, 1 h video, 0.5 h exposición). Planifica 1-2 h diarias para evitar procrastinación.
- **Entregables**:
  - **Repositorio Git**:
    - `README.md`: Instrucciones de ejecución, dependencias, tiempos.
    - `notebook.ipynb`: Código, resultados, preguntas teóricas escritas.
    - `data/nlp_prueba_cc0c2_large.csv`: Dataset generado.
    - `out/`: Gráficos/tablas generadas.
    - `video.mp4`: Video de 5-10 min en formato sprint.
    - `requirements.txt`: Dependencias (Python 3.x, NumPy, Pandas, Matplotlib, opcionalmente NLTK/spaCy, tokenizers, sentence-transformers).
    - `SEEDS_AND_VERSIONS.md`: Semillas (e.g., `random.seed(42)`), versiones de bibliotecas.
    - Mínimo **5 commits** por proyecto con mensajes en español (ejemplo: "Implementar BPE para PC1").
  - **Video (5-10 min, formato sprint)-Sugerencia**:
    - 0:00-0:45: Objetivo, historias de usuario, DoD.
    - 0:45-3:00: Demostración en vivo (Notebook/CLI, ejecución de código).
    - 3:00-4:30: Métricas clave, gráficos, análisis.
    - 4:30-5:00+: Riesgos, next sprint, ubicación de código/datos.
  - **Exposición (10 min + 15 min preguntas)**: Usa Notebook como apoyo. Responde preguntas orales.
- **Restricciones antiplagio**: No uses IA generativa ni copies de internet. Verificación con MOSS/Turnitin y commits. Dataset `nlp_prueba_cc0c2_large.csv` asegura soluciones únicas.
- **Formato del Notebook**:
   -  Objetivo y historias usuario.
   -  Setup reproducible (semillas, versiones).
   -  Implementación (celdas cortas, comentadas).
   -  Experimentos y métricas (tablas/gráficos).
   -  Preguntas teóricas (respuestas escritas, 1-2 párrafos c/u).
   -  Trade-offs, riesgos, next sprint.
   -  Conclusiones técnicas y evidencias.

Se presenta un script genera un dataset ampliado con 10,000 oraciones en español relacionadas con NLP/IA, etiquetadas como 'Positivo', 'Negativo', o 'Neutral'. 
Combina oraciones sintéticas (basadas en plantillas) y ejemplos reales inspirados en el dataset original.  El script usa listas de palabras y estructuras para garantizar diversidad y realismo.

**Dataset**: Usa `nlp_prueba_cc0c2_large.csv` (~10,000 oraciones) generado con el script proporcionado. Descarga desde el enlace del repositorio o genera localmente.

**Ejemplo de `nlp_prueba_cc0c2_large.csv`**
```
Texto,Categoría
La tokenización es clave para procesar texto,Positivo
No entiendo los embeddings vectoriales,Negativo
Los LLMs son impresionantes pero complejos,Neutral
El curso de NLP es fascinante y útil,Positivo
La programación en Python es complicada al principio,Negativo
Entender los embeddings resulta útil en el curso de NLP,Positivo
No entiendo cómo funciona la regularización, es confuso,Negativo
La lematización parece interesante pero fundamental,Neutral
Implementar modelos de lenguaje es innovador en proyectos reales,Positivo
Los transformers son complicados y limitados para datasets pequeños,Negativo
```


### **Proyectos**

#### **Proyecto 1: Clasificación Zero-Shot con Guardrails**
**Temas**: Introducción a NLP/LLMs/modelos fundacionales.  
**Implementación (3 puntos)**: Usa HuggingFace `pipeline` (e.g., `zero-shot-classification`) para clasificar 500 oraciones de `nlp_prueba_cc0c2_large.csv` en 'Positivo', 'Negativo', 'Neutral'. Prueba 2 prompts distintos (e.g., "Clasifica el sentimiento" vs. "Evalúa la emoción"). Implementa un guardrail (regex para nombres propios). Calcula accuracy y matriz de confusión.  
**Teoría (escrita)**:
1. Define modelo fundacional y pretraining.
2. Explica **in-context learning** en zero-shot.
3. Describe riesgos de **prompt injection**.
4. Impacto de tokens en costo computacional.
5. Analiza un fallo de clasificación y solución.  
**Métricas**: Accuracy, matriz de confusión, 5 ejemplos de errores.   
**Entregables**:
    - **Notebook**: Código, métricas, teoría. Commit: "Zero-shot con guardrails PC1".
    - **Video**: Demo de pipeline, resultados, guardrail.  
    - **Exposición**: Presenta resultados, riesgos. 

#### **Proyecto 2: Optimización de carga de datos**
**Temas**: Carga de datos, bibliotecas básicas.  
**Implementación**: Compara Pandas vs. PyTorch DataLoader para cargar `nlp_prueba_cc0c2_large.csv` (10,000 oraciones). Mide tiempo y RAM (con `psutil`) para lectura completa vs. batching (batch_size=32). Usa `collate_fn` para padding simple. Fija semillas.  
**Teoría**:
1. Iteradores vs. listas en memoria.
2. Pros/cons de streaming vs. in-memory.
3. Importancia de semillas en NLP.
4. Define shuffling estable y su impacto.
5. Pseudocódigo de un prefetcher para DataLoader.  
**Métricas**: Tiempo por epoch, RAM pico, estabilidad de batches.  
**Entregables**:
    - **Notebook**: Código, métricas, teoría. Commit: "Carga de datos optimizada PC1".
    - **Video**: Muestra ejecución, métricas.  
    - **Exposición**: Presenta resultados, riesgos. Explica trade-offs. 

#### **Proyecto 3: Tokenización con BPE personalizado**
**Temas**: Tokenización, algoritmo BPE.  
**Implementación**: Entrena BPE (con `tokenizers` o manual) para 5,000 oraciones de `nlp_prueba_cc0c2_large.csv`, generando un vocabulario de 2,000 tokens. Muestra 15 merges y tokenización de 10 oraciones. Compara con `bert-base-multilingual-cased`.  
**Teoría**:
1. Pasos del algoritmo BPE.
2. Impacto del tamaño de vocabulario en OOV.
3. Define subword regularization.
4. BPE vs. WordPiece (diferencias).
5. Rol de BPE en latencia de LLMs.  
**Métricas**: Tamaño de vocabulario, longitud promedio de tokens, ejemplos de segmentación.  
**Entregables**:
    - **Notebook**: Código, merges, teoría. Commit: "BPE personalizado PC1".
    - **Video**: Muestra merges, tokenización.  
    - **Exposición**: Presenta resultados, riesgos. Explica aplicaciones de BPE. 

#### **Proyecto 4: Pipeline de preprocesamiento**
**Temas**: Normalización, lematización, segmentación.  
**Implementación**: Crea 3 pipelines para 5,000 oraciones de `nlp_prueba_cc0c2_large.csv`: (a) crudo, (b) normalizado (minúsculas, sin puntuación), (c) normalizado+lematizado (spaCy en español). Calcula vocabulario y OOV. Analiza 3 oraciones con code-switching (es/en).  
**Teoría**:
1. Stemming vs. lematización.
2. Impacto de segmentación en OOV.
3. Riesgos de normalización agresiva.
4. Estrategias para **code-switching**.
5. Caso donde normalización degrade un modelo.  
**Métricas**: Tamaño de vocabulario, OOV, ejemplos cualitativos.    
**Entregables**:
    - **Notebook**: Código, resultados, teoría. Commit: "Pipeline de preprocesamiento PC1".
    - **Video**: Muestra pipelines, resultados.  
    - **Exposición**: Presenta resultados, riesgos, compara pipelines. 

#### **Proyecto 5: Modelo de lenguaje Trigrama**
**Temas**: Modelos de lenguaje, n-gramas.  
**Implementación**: Implementa un trigrama con Kneser-Ney (o usa `nltk` con explicación de fórmulas) para 8,000 oraciones de `nlp_prueba_cc0c2_large.csv` (train: 6,000, valid: 2,000). Compara con add-1 smoothing en perplejidad. Genera 15 oraciones.  
**Teoría**:
1. Deriva Kneser-Ney (breve).
2. Cross-entropy vs. perplejidad.
3. Manejo de UNK en vocabularios abiertos.
4. Ventajas de Kneser-Ney vs. add-1.
5. Relevancia de n-gramas en 2025.  
**Métricas**: Perplejidad (train/validate), calidad de oraciones.  
**Entregables**:
    - **Notebook**: Código, perplejidad, teoría. Commit: "Trigrama Kneser-Ney PC1".
    - **Video**: Muestra perplejidad, oraciones.  
    - **Exposición**: Presenta resultados, riesgos. Explica smoothing.

#### **Proyecto 6: Clasificación con regularización**
**Temas**: Regularización, evaluación de modelos.  
**Implementación**: Usa regresión logística (scikit-learn) con BoW/TF-IDF para clasificar 8,000 oraciones de `nlp_prueba_cc0c2_large.csv` (Positivo vs. No Positivo). Compara L2 vs. sin regularización. Reporta F1, ROC-AUC, matriz de confusión.  
**Teoría**:
1. Sesgo-varianza en clasificación de texto.
2. F1 vs. ROC-AUC: ¿cuándo usar cada uno?
3. Estrategias para class imbalance.
4. Beneficios de regularización L2.
5. Métricas de **fairness** en este modelo.  
**Métricas**: F1 macro, ROC-AUC, matriz de confusión.  
**Entregables**:
    - **Notebook**: Código, métricas, teoría. Commit: "Clasificación con L2 PC1".
    - **Video**: Muestra entrenamiento, métricas.  
    - **Exposición**: Presenta resultados, riesgos. Explica regularización. 

#### **Proyecto 7: Recuperación de texto con embeddings**
**Temas**: Representaciones de texto, embeddings.  
**Implementación**: Compara TF-IDF vs. embeddings (`sentence-transformers` o promedio de word vectors) para recuperar 10 consultas en 8,000 oraciones de `nlp_prueba_cc0c2_large.csv`. Calcula Recall@10. Visualiza embeddings 2D con PCA.  
**Teoría**:
1. Cosine similarity y normalización L2.
2. Define anisotropía en embeddings.
3. ¿Cuándo TF-IDF supera a embeddings?
4. Curse of dimensionality en recuperación.
5. Impacto de stop-words en TF-IDF.  
**Métricas**: Recall@10, visualización 2D, ejemplos de vecinos.  
**Entregables**:
    - **Notebook**: Código, visualización, teoría. Commit: "Recuperación con embeddings PC1".
    - **Video**: Muestra recuperación, PCA.  
    - **Exposición**: Presenta resultados, riesgos. Compara métodos. 

#### Rúbrica de evaluación (20 puntos por proyecto)

| **Criterio** | **Descripción** | **Puntos** | **Detalles** |
|--------------|-----------------|------------|--------------|
| **Trabajo (Notebook)** | Jupyter Notebook con código, resultados, teoría escrita. | 3 | - **Correctitud funcional (1.5)**: Código ejecuta, usa `nlp_prueba_cc0c2_large.csv`, resultados coherentes. <br> - **Reproducibilidad y organización (1)**: Semillas fijas, versiones en `SEEDS_AND_VERSIONS.md`, estructura clara (`data/`, `out/`), 5+ commits con mensajes en español. <br> - **Teoría escrita (0.5)**: Respuestas a 5 preguntas claras, conectadas a NLP. |
| **Video de ejecución** | Video de 5-10 min en formato sprint. | 4 | - **Guion sprint (1)**: Introduce objetivo, Historía de usuario, DoD (0:00-0:45). <br> - **Demo en vivo (1.5)**: Ejecuta Notebook, muestra código/resultados (0:45-3:00). <br> - **Métricas y análisis (1)**: Interpreta gráficos/tablas, justifica decisiones (3:00-4:30). <br> - **Cierre (0.5)**: Resume riesgos, next sprint, ubicación de código/datos (4:30-5:00+). |
| **Exposición y preguntas** | Presentación (10 min) + preguntas orales (15 min). | 13 | - **Estructura y narrativa (3)**: Presentación clara, usa Notebook, explica implementación/teoría. <br> - **Profundidad técnica (4)**: Demuestra comprensión de conceptos (e.g., BPE, perplejidad). <br> - **Respuesta a preguntas orales (5)**: Responde 2-3 preguntas con precisión (e.g., "¿Cómo mejorarías el modelo?"). <br> - **Visualizaciones (1)**: Gráficos legibles con títulos/ejes. |

**Notas**:
- **Plagio**: Detectado por MOSS, commits inconsistentes, o respuestas genéricas = 0 puntos.
- **Video**: <5 min resta 0.5 puntos. Audio/pantalla poco claros resta 0.5 puntos.
- **Commits**: Menos de 5 commits resta 0.5 puntos. Mensajes genéricos reducen claridad.
- **Exposición**: Falta de preparación (no responder preguntas por ejemplo) impacta  puntos de preguntas orales.

#### Cronograma recomendado (10 días)
Para evitar procrastinación:
- **Día 1**: Crea repositorio, estructura Notebook, genera/instala `nlp_prueba_cc0c2_large.csv`. Commit: "Inicializar PC1".
- **Día 2**: Implementa prototipo básico con dataset. Commit: "Prototipo inicial PC1".
- **Día 3-4**: Completa implementación, calcula métricas. Commit: "Implementación principal PC1".
- **Día 5-6**: Responde preguntas teóricas, genera gráficos en `out/`. Commit: "Teoría y visualizaciones PC1".
- **Día 7**: Graba borrador de video, revisa guion sprint. Commit: "Borrador video PC1".
- **Día 8**: Pule Notebook, verifica reproducibilidad. Commit: "Notebook final PC1".
- **Día 9**: Graba video final, sube a repositorio. Commit: "Video final PC1".
- **Día 10**: Prepara exposición, ensaya respuestas orales. Entrega final.
