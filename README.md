# FAQsBot

# FAQ Bot con Embeddings

Este proyecto es un bot de **Preguntas Frecuentes (FAQ)** que utiliza **embeddings semánticos** y **similitud del coseno** para encontrar la respuesta más similar a una pregunta del usuario. Es un proyecto sencillo creado con fines de práctica en el uso de modelos de `sentence-transformers`.

---

## 📁 Estructura del Proyecto

- `sentence_transformers.py` – Contiene la lógica para generar embeddings y calcular similitudes.
- `fqs.csv` – Archivo CSV que contiene las **preguntas y respuestas** frecuentes.
- `main.py` – Punto de entrada del programa. Captura input del usuario y muestra la respuesta más similar.
- `practice.py` – Archivo de práctica o pruebas con el modelo.
- `config/` – Carpeta para posibles configuraciones adicionales (modelo, parámetros, etc.)

---

## 🧠 ¿Cómo Funciona?

1. El archivo `fqs.csv` contiene pares de **pregunta-respuesta**.
2. El modelo de `sentence-transformers` genera un **embedding** para cada pregunta del CSV.
3. Cuando el usuario escribe una pregunta, se genera su embedding.
4. Se calcula la **similitud del coseno** entre el embedding del usuario y los embeddings del CSV.
5. Se retorna la pregunta más similar y su respuesta correspondiente.

---

## 🚀 Requisitos

- Python 3.8+
- `pandas`
- `sentence-transformers`
- `scikit-learn` (opcional para cálculo de similitud)

Instalación recomendada:

```bash
pip install -r requirements.txt
```

Ejemplo de un `requirements.txt` básico:

```
pandas
sentence-transformers
scikit-learn
```

---

## ▶️ Uso

Ejecuta el archivo principal:

```bash
python main.py
```

Luego, escribe una pregunta y el bot te devolverá la más parecida del CSV con su respuesta.

---

## 💡 Ejemplo

**Input del usuario:**

```
¿Cómo puedo cambiar mi contraseña?
```

**Respuesta generada:**

> **Pregunta similar:** "¿Dónde cambio mi contraseña?"
>
> **Respuesta:** "Puedes cambiar tu contraseña desde la configuración de tu cuenta, en la sección 'Seguridad'."

---

## ✍️ Autor

**Gean Franco Jacome Laguna**

---

## 📜 Licencia

Este proyecto es solo para fines educativos y de práctica. No se especifica una licencia formal.
