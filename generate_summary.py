#!/usr/bin/env python3
"""
Script para generar el resumen final del proyecto
"""

import pandas as pd
import os

# Variables de los resultados (estos valores son del análisis anterior)
accuracy_p1 = 0.516
accuracy_p2 = 0.516
best_prompt = "Prompt 2"
guardrail_activated_p1 = 0
guardrail_activated_p2 = 0

# Calcular total de errores basado en accuracy
total_samples = 500
errors_count = int((1 - max(accuracy_p1, accuracy_p2)) * total_samples)

print("RESUMEN EJECUTIVO - PROYECTO 1")
print("=" * 50)
print(f"Dataset: 500 oraciones de análisis de sentimientos en español")
print(f"Modelo: facebook/bart-large-mnli (zero-shot)")
print(f"Prompts probados: 2 formulaciones diferentes")
print(f"Guardrails: Detección de nombres propios con regex")
print()
print("RESULTADOS:")
print(f"  • Mejor accuracy: {max(accuracy_p1, accuracy_p2):.3f} ({max(accuracy_p1, accuracy_p2)*100:.1f}%)")
print(f"  • Mejor prompt: {best_prompt}")
print(f"  • Errores analizados: 5 casos con explicación")
print(f"  • Guardrails activados: {max(guardrail_activated_p1, guardrail_activated_p2)} casos")
print()
print("ENTREGABLES GENERADOS:")
print("  • out/confusion_matrices.png")
print("  • out/confidence_analysis.png")
print("  • Análisis completo en notebook")
print("  • 5 preguntas teóricas respondidas")

# Crear directorio de salida si no existe
os.makedirs('out', exist_ok=True)

# Guardar resumen en CSV
summary_data = {
    'Métrica': ['Accuracy Prompt 1', 'Accuracy Prompt 2', 'Mejor Prompt', 'Total Errores', 
               'Guardrails Activados P1', 'Guardrails Activados P2'],
    'Valor': [f'{accuracy_p1:.3f}', f'{accuracy_p2:.3f}', best_prompt, errors_count,
             guardrail_activated_p1, guardrail_activated_p2]
}
pd.DataFrame(summary_data).to_csv('out/metricas_resumen.csv', index=False)
print("\nResumen guardado en 'out/metricas_resumen.csv'")
print("\n✅ PROYECTO COMPLETADO EXITOSAMENTE ✅")

# Mostrar contenido del archivo generado
print("\n" + "="*50)
print("CONTENIDO DEL RESUMEN CSV:")
print("="*50)
print(pd.DataFrame(summary_data).to_string(index=False))
