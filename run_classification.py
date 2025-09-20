#!/usr/bin/env python3
"""
Proyecto 1: Clasificación Zero-Shot con Guardrails
Script para ejecutar el análisis completo
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
print("PROYECTO 1: CLASIFICACIÓN ZERO-SHOT CON GUARDRAILS")
print("=" * 60)

# 1. Cargar datos
print("1. Cargando datos...")
df_full = pd.read_csv('data/nlp_prueba_cc0c2_large.csv')
print(f"   Dataset completo: {len(df_full)} oraciones")

# Seleccionar 500 oraciones
df_sample = df_full.groupby('Categoría').apply(
    lambda x: x.sample(n=min(167, len(x)), random_state=SEED)
).reset_index(drop=True)
df_sample = df_sample.sample(n=500, random_state=SEED).reset_index(drop=True)

print(f"   Muestra para análisis: {len(df_sample)} oraciones")
print(f"   Distribución: {dict(df_sample['Categoría'].value_counts())}")

# 2. Configurar modelo
print("\n2. Configurando modelo zero-shot...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    return_all_scores=True
)

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

# Test guardrail
test_text = "María piensa que Python es complicado"
proper_nouns = detect_proper_nouns(test_text)
print(f"   Prueba guardrail:")
print(f"   Texto: '{test_text}'")
print(f"   Nombres propios detectados: {proper_nouns}")

# 4. Ejecutar clasificación
print("\n4. Ejecutando clasificación...")
print("   Esto puede tomar 5-10 minutos para 500 oraciones...")

results_prompt1 = []
results_prompt2 = []

# Clasificar con progreso
for i, row in df_sample.iterrows():
    if i % 50 == 0:
        progress = (i+1)/500*100
        print(f"   Procesando: {i+1}/500 oraciones ({progress:.1f}%)")
    
    text = row['Texto']
    
    # Prompt 1
    result1 = classify_with_prompts(text, PROMPT_1, 1)
    results_prompt1.append(result1)
    
    # Prompt 2  
    result2 = classify_with_prompts(text, PROMPT_2, 2)
    results_prompt2.append(result2)

print("   ✅ Clasificación completada")

# 5. Analizar resultados
print("\n5. Analizando resultados...")

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

print(f"\n   ACCURACY COMPARISON:")
print(f"   Prompt 1 ('sentimiento'): {accuracy_p1:.3f} ({accuracy_p1*100:.1f}%)")
print(f"   Prompt 2 ('emoción'):     {accuracy_p2:.3f} ({accuracy_p2*100:.1f}%)")
print(f"   Diferencia: {abs(accuracy_p1-accuracy_p2):.3f}")

best_prompt = "Prompt 1" if accuracy_p1 > accuracy_p2 else "Prompt 2"
best_pred_col = 'pred_prompt1' if accuracy_p1 > accuracy_p2 else 'pred_prompt2'
print(f"   Mejor performance: {best_prompt}")

# 6. Generar visualizaciones
print("\n6. Generando visualizaciones...")

# Crear directorio out si no existe
os.makedirs('out', exist_ok=True)

# Matrices de confusión
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm1 = confusion_matrix(df_results['Categoría'], df_results['pred_prompt1'], 
                       labels=['Positivo', 'Negativo', 'Neutral'])
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positivo', 'Negativo', 'Neutral'],
            yticklabels=['Positivo', 'Negativo', 'Neutral'])
plt.title(f'Prompt 1: "sentimiento"\nAccuracy: {accuracy_p1:.3f}')
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')

plt.subplot(1, 2, 2)
cm2 = confusion_matrix(df_results['Categoría'], df_results['pred_prompt2'],
                       labels=['Positivo', 'Negativo', 'Neutral'])
sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Positivo', 'Negativo', 'Neutral'],
            yticklabels=['Positivo', 'Negativo', 'Neutral'])
plt.title(f'Prompt 2: "emoción"\nAccuracy: {accuracy_p2:.3f}')
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')

plt.tight_layout()
plt.savefig('out/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Matrices de confusión guardadas en 'out/confusion_matrices.png'")

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

# 8. Análisis de errores
print("\n8. Analizando errores...")

errors = df_results[df_results['Categoría'] != df_results[best_pred_col]].copy()
conf_col = 'conf_prompt1' if best_pred_col == 'pred_prompt1' else 'conf_prompt2'
errors = errors.sort_values(conf_col, ascending=False)

print(f"   Total de errores: {len(errors)} de 500 ({len(errors)/500*100:.1f}%)")
print(f"\n   TOP 5 ERRORES (mayor confianza en predicción incorrecta):")

for i, (idx, row) in enumerate(errors.head(5).iterrows()):
    print(f"\n   Error #{i+1}:")
    print(f"     Texto: '{row['Texto']}'")
    print(f"     Real: {row['Categoría']} | Predicho: {row[best_pred_col]}")
    print(f"     Confianza: {row[conf_col]:.3f}")

# 9. Reportes de clasificación
print("\n9. Generando reportes detallados...")

print(f"\n   CLASSIFICATION REPORT - {best_prompt.upper()}:")
print(classification_report(df_results['Categoría'], df_results[best_pred_col]))

# 10. Guardar resultados
print("\n10. Guardando resultados...")

# Guardar DataFrame completo
df_results.to_csv('out/resultados_completos.csv', index=False)

# Guardar resumen de métricas
summary_data = {
    'Métrica': ['Accuracy Prompt 1', 'Accuracy Prompt 2', 'Mejor Prompt', 'Total Errores', 
               'Guardrails Activados P1', 'Guardrails Activados P2'],
    'Valor': [f'{accuracy_p1:.3f}', f'{accuracy_p2:.3f}', best_prompt, len(errors),
             guardrail_activated_p1, guardrail_activated_p2]
}
pd.DataFrame(summary_data).to_csv('out/metricas_resumen.csv', index=False)

print("   ✅ Resultados guardados en:")
print("      - out/resultados_completos.csv")
print("      - out/metricas_resumen.csv")
print("      - out/confusion_matrices.png")

print("\n" + "=" * 60)
print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
print(f"   Mejor accuracy: {max(accuracy_p1, accuracy_p2):.3f} ({max(accuracy_p1, accuracy_p2)*100:.1f}%)")
print(f"   Mejor prompt: {best_prompt}")
print("=" * 60)
