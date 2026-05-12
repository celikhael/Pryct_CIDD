import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import numpy as np

from src.train import train_model
from src.evaluate import evaluate_model
from src.graphs import show_model_graphs


# ─── Columnas de entrada (sin id ni diagnosis) ───────────────────────────────
FEATURE_COLS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

# Ejemplo maligno — ID 842302 (M)
EXAMPLE_MALIGNANT = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
    0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
    0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
]

# Ejemplo benigno — ID 857637 (B)
EXAMPLE_BENIGN = [
    9.029, 17.33, 58.79, 250.5, 0.1066, 0.1413, 0.313, 0.04375,
    0.2111, 0.08046, 0.3274, 1.194, 1.885, 17.67, 0.009549, 0.08606,
    0.3038, 0.0646, 0.02675, 0.01737, 9.956, 21.87, 63.62, 292.1,
    0.1696, 0.4244, 0.9454, 0.2112, 0.2882, 0.1155,
]


# ─── Acciones principales ─────────────────────────────────────────────────────

def entrenar_modelo():
    try:
        train_model()
        resultado_label.config(text="Modelo entrenado correctamente", fg="#1b5e20")
        messagebox.showinfo("Éxito", "El modelo fue entrenado y guardado correctamente.")
    except Exception as e:
        resultado_label.config(text="Error al entrenar el modelo.", fg="#b71c1c")
        messagebox.showerror("Error", f"Ocurrió un problema al entrenar el modelo:\n{e}")


def evaluar_modelo():
    try:
        accuracy = evaluate_model()
        resultado_label.config(text=f"Accuracy del modelo: {accuracy:.2f}%", fg="#0d47a1")
        messagebox.showinfo("Evaluación completada", f"Precisión del modelo: {accuracy:.2f}%")
    except Exception as e:
        resultado_label.config(text="Error al evaluar el modelo.", fg="#b71c1c")
        messagebox.showerror("Error", f"Ocurrió un problema al evaluar el modelo:\n{e}")


def mostrar_graficas():
    try:
        show_model_graphs()
        resultado_label.config(text="Gráficas generadas correctamente", fg="#1b5e20")
    except Exception as e:
        resultado_label.config(text="Error al generar las gráficas.", fg="#b71c1c")
        messagebox.showerror("Error", f"Ocurrió un problema al generar las gráficas:\n{e}")


# ─── Ventana de predicción ────────────────────────────────────────────────────

def abrir_ventana_prediccion():
    pred_win = tk.Toplevel(ventana)
    pred_win.title("Predicción de Diagnóstico")
    pred_win.geometry("780x640")
    pred_win.resizable(True, True)
    pred_win.configure(bg="#f3f4f6")

    # ── Título ──
    tk.Label(
        pred_win,
        text="Predicción de Diagnóstico",
        font=("Helvetica", 16, "bold"),
        bg="#f3f4f6", fg="#1c2733",
    ).pack(pady=(16, 2))

    tk.Label(
        pred_win,
        text="Ingresa los valores de cada atributo y presiona «Predecir».",
        font=("Helvetica", 10),
        bg="#f3f4f6", fg="#4a4a4a",
    ).pack(pady=(0, 10))

    # ── Área desplazable ──
    outer = tk.Frame(pred_win, bg="#f3f4f6")
    outer.pack(fill="both", expand=True, padx=16)

    canvas = tk.Canvas(outer, bg="#f3f4f6", highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    scroll_frame = tk.Frame(canvas, bg="#f3f4f6")
    win_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def on_canvas_resize(event):
        canvas.itemconfig(win_id, width=event.width)

    scroll_frame.bind("<Configure>", on_configure)
    canvas.bind("<Configure>", on_canvas_resize)

    # Scroll con rueda del ratón
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # ── Grid de campos ──
    entries = {}
    COLS = 3

    for idx, col in enumerate(FEATURE_COLS):
        row_idx = idx // COLS
        col_idx = idx % COLS

        cell = tk.Frame(scroll_frame, bg="#f3f4f6", padx=6, pady=4)
        cell.grid(row=row_idx, column=col_idx, sticky="ew", padx=4, pady=2)
        scroll_frame.columnconfigure(col_idx, weight=1)

        tk.Label(
            cell,
            text=col,
            font=("Helvetica", 9, "bold"),
            bg="#f3f4f6", fg="#374151",
            anchor="w",
        ).pack(fill="x")

        entry = tk.Entry(
            cell,
            font=("Helvetica", 10),
            relief="solid", bd=1,
            highlightthickness=1,
            highlightcolor="#1976d2",
        )
        entry.insert(0, str(EXAMPLE_VALUES[idx]))
        entry.pack(fill="x", ipady=3)
        entries[col] = entry

    # ── Botones ──
    btn_frame = tk.Frame(pred_win, bg="#f3f4f6")
    btn_frame.pack(pady=10)

    def rellenar_ejemplo(valores):
        for col, val in zip(FEATURE_COLS, valores):
            entries[col].delete(0, tk.END)
            entries[col].insert(0, str(val))

    def limpiar_campos():
        for entry in entries.values():
            entry.delete(0, tk.END)

    def realizar_prediccion():
        try:
            model = joblib.load("model/modelo.pkl")
        except FileNotFoundError:
            messagebox.showerror(
                "Modelo no encontrado",
                "No se encontró el archivo model/modelo.pkl\n"
                "Por favor entrena el modelo primero.",
            )
            return

        # Leer y validar valores
        try:
            valores = [float(entries[col].get()) for col in FEATURE_COLS]
        except ValueError:
            messagebox.showerror(
                "Valor inválido",
                "Todos los campos deben contener números.\n"
                "Revisa que no haya campos vacíos o con texto.",
            )
            return

        X_input = np.array(valores).reshape(1, -1)

        try:
            prediccion = model.predict(X_input)[0]
            probabilidad = model.predict_proba(X_input)[0]
        except Exception as e:
            messagebox.showerror("Error en predicción", str(e))
            return

        # Interpretar resultado
        es_maligno = prediccion == 1 or str(prediccion).upper() == "M"
        etiqueta = "MALIGNO (M)" if es_maligno else "BENIGNO (B)"
        prob_m = probabilidad[1] * 100 if len(probabilidad) > 1 else float("nan")
        prob_b = probabilidad[0] * 100

        color = "#b71c1c" if es_maligno else "#1b5e20"
        icono = "" if es_maligno else ""

        # Ventana de resultado
        res_win = tk.Toplevel(pred_win)
        res_win.title("Resultado de Predicción")
        res_win.geometry("420x260")
        res_win.resizable(False, False)
        res_win.configure(bg="#ffffff")

        tk.Label(
            res_win,
            text=f"{icono}  Resultado del Diagnóstico",
            font=("Helvetica", 14, "bold"),
            bg="#ffffff", fg="#1c2733",
        ).pack(pady=(24, 6))

        tk.Label(
            res_win,
            text=etiqueta,
            font=("Helvetica", 28, "bold"),
            bg="#ffffff", fg=color,
        ).pack(pady=4)

        tk.Label(
            res_win,
            text=f"Probabilidad  Maligno: {prob_m:.1f}%   |   Benigno: {prob_b:.1f}%",
            font=("Helvetica", 11),
            bg="#ffffff", fg="#374151",
        ).pack(pady=8)

        # Barra de probabilidad
        bar_frame = tk.Frame(res_win, bg="#ffffff")
        bar_frame.pack(pady=4, padx=40, fill="x")

        bar_canvas = tk.Canvas(bar_frame, height=20, bg="#e5e7eb", highlightthickness=0)
        bar_canvas.pack(fill="x")
        bar_canvas.update_idletasks()
        w = bar_canvas.winfo_width() or 340
        fill_w = int(w * prob_m / 100)
        bar_canvas.create_rectangle(0, 0, fill_w, 20, fill="#b71c1c", outline="")
        bar_canvas.create_rectangle(fill_w, 0, w, 20, fill="#1b5e20", outline="")

        tk.Button(
            res_win,
            text="Cerrar",
            font=("Helvetica", 11, "bold"),
            bg="#1976d2", fg="white",
            activebackground="#1565c0", activeforeground="white",
            relief="flat", width=12, height=1,
            command=res_win.destroy,
        ).pack(pady=16)

    tk.Button(
        btn_frame,
        text="Ejemplo Maligno",
        font=("Helvetica", 11, "bold"),
        bg="#b71c1c", fg="white",
        activebackground="#7f0000", activeforeground="white",
        width=16, height=2, relief="flat",
        command=lambda: rellenar_ejemplo(EXAMPLE_MALIGNANT),
    ).grid(row=0, column=0, padx=8, pady=4)

    tk.Button(
        btn_frame,
        text="Ejemplo Benigno",
        font=("Helvetica", 11, "bold"),
        bg="#2e7d32", fg="white",
        activebackground="#1b5e20", activeforeground="white",
        width=16, height=2, relief="flat",
        command=lambda: rellenar_ejemplo(EXAMPLE_BENIGN),
    ).grid(row=0, column=1, padx=8, pady=4)

    tk.Button(
        btn_frame,
        text="🗑  Limpiar",
        font=("Helvetica", 11, "bold"),
        bg="#546e7a", fg="white",
        activebackground="#455a64", activeforeground="white",
        width=16, height=2, relief="flat",
        command=limpiar_campos,
    ).grid(row=1, column=0, padx=8, pady=4)

    tk.Button(
        btn_frame,
        text="🔍  Predecir",
        font=("Helvetica", 11, "bold"),
        bg="#1976d2", fg="white",
        activebackground="#1565c0", activeforeground="white",
        width=16, height=2, relief="flat",
        command=realizar_prediccion,
    ).grid(row=1, column=1, padx=8, pady=4)


# ─── Ventana principal ────────────────────────────────────────────────────────

ventana = tk.Tk()
ventana.title("Clasificador de Diagnóstico")
ventana.geometry("560x460")
ventana.resizable(False, False)
ventana.configure(bg="#f3f4f6")

titulo = tk.Label(
    ventana,
    text="Clasificador de Diagnóstico",
    font=("Helvetica", 18, "bold"),
    bg="#f3f4f6", fg="#1c2733",
)
titulo.pack(pady=(20, 10))

subtitulo = tk.Label(
    ventana,
    text="Entrena, evalúa y visualiza el comportamiento del modelo.",
    font=("Helvetica", 11),
    bg="#f3f4f6", fg="#4a4a4a",
)
subtitulo.pack(pady=(0, 20))

botones_frame = tk.Frame(ventana, bg="#f3f4f6")
botones_frame.pack()

btn_entrenar = tk.Button(
    botones_frame,
    text="Entrenar Modelo",
    font=("Helvetica", 12, "bold"),
    bg="#1976d2", fg="white",
    activebackground="#1565c0", activeforeground="white",
    width=18, height=2, relief="flat",
    command=entrenar_modelo,
)
btn_entrenar.grid(row=0, column=0, padx=10, pady=8)

btn_evaluar = tk.Button(
    botones_frame,
    text="Evaluar Precisión",
    font=("Helvetica", 12, "bold"),
    bg="#388e3c", fg="white",
    activebackground="#2e7d32", activeforeground="white",
    width=18, height=2, relief="flat",
    command=evaluar_modelo,
)
btn_evaluar.grid(row=0, column=1, padx=10, pady=8)

btn_graficas = tk.Button(
    botones_frame,
    text="Mostrar Gráficas",
    font=("Helvetica", 12, "bold"),
    bg="#f57c00", fg="white",
    activebackground="#ef6c00", activeforeground="white",
    width=18, height=2, relief="flat",
    command=mostrar_graficas,
)
btn_graficas.grid(row=1, column=0, padx=10, pady=8)

btn_predecir = tk.Button(
    botones_frame,
    text="🔍 Predecir Caso",
    font=("Helvetica", 12, "bold"),
    bg="#6a1b9a", fg="white",
    activebackground="#4a148c", activeforeground="white",
    width=18, height=2, relief="flat",
    command=abrir_ventana_prediccion,
)
btn_predecir.grid(row=1, column=1, padx=10, pady=8)

resultado_label = tk.Label(
    ventana,
    text="Esperando acción...",
    font=("Helvetica", 12),
    bg="#eef3f8", fg="#0d47a1",
    width=45, height=2,
    relief="ridge", bd=2,
)
resultado_label.pack(pady=(20, 0))
