# Challenge-DS-LATAM-TelecomX-P2
## 🎯 Objetivo de la Parte 2

1) **Preparar un dataset modelable** (todas las variables numéricas, sin fuga de información).  
2) **Enfrentar el desbalance** con técnicas adecuadas (SMOTE).  
3) **Entrenar y comparar** múltiples clasificadores para predecir churn.  
4) **Evaluar** con métricas robustas (F1, ROC‑AUC, PR‑Curve).  
5) **Interpretar** los modelos (coeficientes/feature importance) para decisiones de negocio.  
6) **Concluir** con acciones de retención y un plan de despliegue.

---

## 📂 Dataset

- `datos_preparados.csv` — Base **limpia** con categóricas en formato `Yes/No` y otras categorías nominales; `Churn` como `Yes/No` o 0/1.

> Asegúrate de **eliminar `customerID`** antes del modelado (no aporta y puede sesgar).

---

## 🧭 Flujo (Roadmap ML)

**Preparación**
- **Codificación OHE** con `pandas.get_dummies` → `datos_codificados` (todas las categóricas a numérico; `Churn` excluida de OHE).  
- **Split 70/30** estratificado: `X_train_full`, `X_test_full`, `y_train`, `y_test`.  
- **StandardScaler**: solo para modelos sensibles a escala (LogReg/KNN).  
- **Correlaciones** (en train; normalizadas por requerimiento): ranking de asociación contra `Churn_bin`.  
- **Baseline** con `DummyClassifier (most_frequent)` para piso mínimo de desempeño.  
- **Top‑N features** por |correlación| (definidas **en train**, sin fuga).  
  - Conjuntos finales listos:  
    - **Escalados** → `X_train_scaled_top`, `X_test_scaled_top` (LogReg/KNN).  
    - **Sin escalar** → `X_train_tree_top`, `X_test_tree_top` (RF/XGB).

**Balanceo**
- **SMOTE** aplicado **solo en entrenamiento** (evita fuga), en dos espacios:  
  - `X_train_scaled_top_sm, y_train_scaled_sm` → LogReg/KNN.  
  - `X_train_tree_top_sm,  y_train_tree_sm`  → RF/XGB.

**Entrenamiento y comparación**
- Modelos: **Regresión Logística**, **KNN**, **Random Forest**, **XGBoost**.  
- Métricas en test: **Accuracy**, **Precision**, **Recall**, **F1**, **ROC‑AUC**.  
- Visuales comparativos: **Matrices de confusión**, **ROC** y **Precision‑Recall** (`RocCurveDisplay`, `PrecisionRecallDisplay`).

**Tuning + Validación (sin fuga)**
- Pipelines con **SMOTE/escala dentro de CV** (`StratifiedKFold` 5‑fold).  
- Búsqueda de hiperparámetros (Grid/Random).  
- **Diagnóstico 4.3**: comparar **F1 (CV)** vs **F1 (test)** para detectar **overfitting** / **underfitting**.

**Interpretación**
- **LogReg**: coeficientes (signo/dirección).  
- **RF/XGB**: `feature_importances_`.  
- **KNN**: **Permutation Importance** (∆F1 al permutar).  
- **Consenso**: normalizar importancias y promediarlas para un ranking unificado.

**Conclusiones estratégicas**
- KPI: churn global. Drivers frecuentes: **contrato mensual**, **sin soporte/seguridad**, **método de pago e‑check**, **tenure bajo**, **cargos altos**.  
- Acciones: **migración a contratos anuales**, **bundles con soporte**, **incentivos de método de pago**, **onboarding 0–3 meses**, **step‑up pricing**.  
- Roadmap de despliegue (4 semanas) con pilotos y control.

---

## 🛠️ Requisitos (librerías y versiones)

Probado con:

- `pandas==2.2.2`  
- `numpy==2.0.2`  
- `matplotlib==3.10.0`  
- `seaborn==0.13.2`  
- `scikit-learn==1.6.1`  
- **Adicionales**: `imbalanced-learn` (SMOTE), `xgboost`

### Instalación rápida
```bash
pip install pandas==2.2.2 numpy==2.0.2 matplotlib==3.10.0 seaborn==0.13.2 scikit-learn==1.6.1
pip install imbalanced-learn xgboost
```

### Verificar versiones del runtime
```python
import importlib, platform
for pkg in ["pandas","numpy","matplotlib","seaborn","sklearn","imblearn","xgboost"]:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}=={getattr(m,'__version__','N/D')}")
    except Exception:
        print(f"{pkg}==NO_INSTALADO")
print("Python:", platform.python_version())
```

---

## ▶️ Ejecución (paso a paso)

1) Coloca `datos_preparados.csv` junto al notebook   
2) Corre las celdas en orden:  
   - OHE → Split 70/30 → Scaler → Correlaciones → Baseline → Top‑N features → SMOTE → Modelos → Tuning + CV → Interpretación → Conclusiones.  
3) Usa las curvas **ROC/PR** para comparar; elige el **modelo principal** según objetivo (Recall/F1 para retención, ROC‑AUC para ranking).  
4) Renderiza el **Informe de Conclusiones** (celda Markdown incluida en el notebook).

---

## 📊 Salidas esperadas

- Tabla `results_df` con métricas de test.  
- Gráficos: **confusión**, **ROC**, **Precision‑Recall**.  
- Tabla de **CV vs Test** (diagnóstico de over/underfitting).  
- Gráficos de **importancias** y **consenso**.  
- **Informe de Conclusiones Estratégicas** (Markdown).

---

## 🔒 Buenas prácticas implementadas

- **Sin fuga**: selección de features y correlaciones sólo con **train**; SMOTE/escala **dentro de CV**.  
- **Desbalance controlado**: SMOTE aplicado **sólo en train**.  
- **Trazabilidad**: baseline, modelos comparables y visuales estándar.  
- **Interpretabilidad**: coeficientes, importancias, permutación y consenso.

---

## 🧱 Estructura recomendada

```
.
├── TelecomX_P2_ML_Colab.ipynb     # Notebook PARTE 2 (ML)
├── datos_preparados.csv           # Dataset de entrada
└── README.md                      # Este documento
```



