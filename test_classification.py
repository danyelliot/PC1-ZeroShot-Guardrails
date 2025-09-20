#!/usr/bin/env python3
"""
Proyecto 1: Clasificación Zero-Shot con Guardrails - Version de Prueba
Script para ejecutar un análisis rápido con una muestra pequeña
"""

import random
import numpy as np
import pandas as pd
import re
import warnings
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings('ignore')

# Configuración reproducible
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("PROYECTO 1: CLASIFICACIÓN ZERO-SHOT - PRUEBA RÁPIDA")
print("=" * 60)

# 1. Cargar datos - solo 50 muestras para prueba
print("1. Cargando datos...")
df_full = pd.read_csv('data/nlp_prueba_cc0c2_large.csv')
print(f"   Dataset completo: {len(df_full)} oraciones")

# Seleccionar solo 50 oraciones para prueba rápida
df_sample = df_full.groupby('Categoría').apply(
    lambda x: x.sample(n=min(17, len(x)), random_state=SEED)
).reset_index(drop=True)
df_sample = df_sample.sample(n=50, random_state=SEED).reset_index(drop=True)

print(f"   Muestra para análisis: {len(df_sample)} oraciones")
print(f"   Distribución: {dict(df_sample['Categoría'].value_counts())}")

# 2. Configurar modelo
print("\n2. Configurando modelo zero-shot...")
try:
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        return_all_scores=True
    )
    print("   ✅ Pipeline configurado correctamente")
except Exception as e:
    print(f"   ❌ Error configurando pipeline: {e}")
    exit(1)

# Prompts
PROMPT_1 = ["sentimiento positivo", "sentimiento negativo", "sentimiento neutral"]
PROMPT_2 = ["emoción positiva", "emoción negativa", "emoción neutral"]

print(f"   Prompt 1: {PROMPT_1}")
print(f"   Prompt 2: {PROMPT_2}")

# 3. Implementar guardrails
print("\n3. Implementando guardrails...")

def detect_proper_nouns(text):
    """Detecta nombres propios usando regex"""
    pattern = r'\b(?<!^)(?<!\. )[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b'
    matches = re.findall(pattern, text)
    return matches

def apply_guardrail(text, prediction, confidence):
    """Aplica guardrail y ajusta predicción si es necesario"""
    proper_nouns = detect_proper_nouns(text)
    
    if proper_nouns:
        if confidence < 0.6:
            return "Neutral", f"Guardrail activado: nombres propios detectados {proper_nouns}, baja confianza"
        else:
            return prediction, f"Nombres propios detectados {proper_nouns}, pero alta confianza"
    
    return prediction, "Sin activación de guardrail"

def map_prediction_to_category(prediction, prompt_type):
    """Mapea las predicciones del modelo a nuestras categorías"""
    if prompt_type == 1:
        mapping = {
            "sentimiento positivo": "Positivo",
            "sentimiento negativo": "Negativo", 
            "sentimiento neutral": "Neutral"
        }
    else:
        mapping = {
            "emoción positiva": "Positivo",
            "emoción negativa": "Negativo",
            "emoción neutral": "Neutral"
        }
    return mapping.get(prediction, prediction)

def classify_with_prompts(text, prompt_labels, prompt_num):
    """Clasifica un texto usando el prompt especificado"""
    try:
        result = classifier(text, prompt_labels)
        
        # Obtener la predicción con mayor score
        best_label = result['labels'][0]
        best_score = result['scores'][0]
        
        # Mapear a nuestras categorías
        mapped_category = map_prediction_to_category(best_label, prompt_num)
        
        # Aplicar guardrail
        final_prediction, guardrail_msg = apply_guardrail(text, mapped_category, best_score)
        
        return {
            'prediction': final_prediction,
            'confidence': best_score,
            'original_label': best_label,
            'guardrail_msg': guardrail_msg,
            'all_scores': dict(zip(result['labels'], result['scores']))
        }
    except Exception as e:
        print(f"   Error clasificando '{text}': {e}")
        return {
            'prediction': 'Neutral',
            'confidence': 0.0,
            'original_label': 'error',
            'guardrail_msg': f'Error: {e}',
            'all_scores': {}
        }

# Test guardrail
test_text = "María piensa que Python es complicado"
proper_nouns = detect_proper_nouns(test_text)
print(f"   Prueba guardrail:")
print(f"   Texto: '{test_text}'")
print(f"   Nombres propios detectados: {proper_nouns}")

# Test de clasificación simple
print("\n4. Probando clasificación...")
test_result = classify_with_prompts("Python es fascinante", PROMPT_1, 1)
print(f"   Test: {test_result}")

# 5. Ejecutar clasificación en muestra pequeña
print("\n5. Ejecutando clasificación en 50 muestras...")

results_prompt1 = []
results_prompt2 = []

# Clasificar con progreso
for i, row in df_sample.iterrows():
    if i % 10 == 0:
        progress = (i+1)/50*100
        print(f"   Procesando: {i+1}/50 oraciones ({progress:.1f}%)")
    
    text = row['Texto']
    
    # Prompt 1
    result1 = classify_with_prompts(text, PROMPT_1, 1)
    results_prompt1.append(result1)
    
    # Prompt 2  
    result2 = classify_with_prompts(text, PROMPT_2, 2)
    results_prompt2.append(result2)

print("   ✅ Clasificación completada")

# 6. Analizar resultados
print("\n6. Analizando resultados...")

# Crear DataFrame con resultados
df_results = df_sample.copy()
df_results['pred_prompt1'] = [r['prediction'] for r in results_prompt1]
df_results['conf_prompt1'] = [r['confidence'] for r in results_prompt1]
df_results['guardrail_prompt1'] = [r['guardrail_msg'] for r in results_prompt1]

df_results['pred_prompt2'] = [r['prediction'] for r in results_prompt2]
df_results['conf_prompt2'] = [r['confidence'] for r in results_prompt2]
df_results['guardrail_prompt2'] = [r['guardrail_msg'] for r in results_prompt2]

# Calcular accuracy
accuracy_p1 = accuracy_score(df_results['Categoría'], df_results['pred_prompt1'])
accuracy_p2 = accuracy_score(df_results['Categoría'], df_results['pred_prompt2'])

print(f"\n   ACCURACY COMPARISON (muestra de 50):")
print(f"   Prompt 1 ('sentimiento'): {accuracy_p1:.3f} ({accuracy_p1*100:.1f}%)")
print(f"   Prompt 2 ('emoción'):     {accuracy_p2:.3f} ({accuracy_p2*100:.1f}%)")
print(f"   Diferencia: {abs(accuracy_p1-accuracy_p2):.3f}")

best_prompt = "Prompt 1" if accuracy_p1 > accuracy_p2 else "Prompt 2"
best_pred_col = 'pred_prompt1' if accuracy_p1 > accuracy_p2 else 'pred_prompt2'
print(f"   Mejor performance: {best_prompt}")

# 7. Análisis de guardrails
print("\n7. Analizando guardrails...")

guardrail_activated_p1 = df_results['guardrail_prompt1'].str.contains('Guardrail activado').sum()
guardrail_activated_p2 = df_results['guardrail_prompt2'].str.contains('Guardrail activado').sum()

proper_nouns_detected_p1 = df_results['guardrail_prompt1'].str.contains('Nombres propios detectados').sum()
proper_nouns_detected_p2 = df_results['guardrail_prompt2'].str.contains('Nombres propios detectados').sum()

print(f"   Prompt 1 - Guardrails activados: {guardrail_activated_p1} casos")
print(f"   Prompt 1 - Nombres propios detectados: {proper_nouns_detected_p1} casos")
print(f"   Prompt 2 - Guardrails activados: {guardrail_activated_p2} casos")
print(f"   Prompt 2 - Nombres propios detectados: {proper_nouns_detected_p2} casos")

# 8. Mostrar algunos ejemplos
print("\n8. Ejemplos de clasificación:")
for i, row in df_results.head(5).iterrows():
    print(f"\n   Ejemplo {i+1}:")
    print(f"     Texto: '{row['Texto']}'")
    print(f"     Real: {row['Categoría']}")
    print(f"     Pred P1: {row['pred_prompt1']} ({row['conf_prompt1']:.3f})")
    print(f"     Pred P2: {row['pred_prompt2']} ({row['conf_prompt2']:.3f})")

# 9. Crear directorio de salida
os.makedirs('out', exist_ok=True)

# 10. Guardar resultados de prueba
df_results.to_csv('out/resultados_prueba.csv', index=False)

print("\n" + "=" * 60)
print("✅ PRUEBA COMPLETADA EXITOSAMENTE")
print(f"   Muestra: 50 oraciones")
print(f"   Mejor accuracy: {max(accuracy_p1, accuracy_p2):.3f} ({max(accuracy_p1, accuracy_p2)*100:.1f}%)")
print(f"   Mejor prompt: {best_prompt}")
print("   Resultados guardados en 'out/resultados_prueba.csv'")
print("=" * 60)
