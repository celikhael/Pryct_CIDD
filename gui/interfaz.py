import tkinter as tk
from tkinter import messagebox

from src.train import train_model
from src.evaluate import evaluate_model
from src.graphs import show_model_graphs


def entrenar_modelo():
    try:
        train_model()
        resultado_label.config(
            text="Modelo entrenado correctamente",
            fg="#1b5e20"
        )
        messagebox.showinfo(
            "Exito",
            "El modelo fue entrenado y guardado correctamente."
        )

    except Exception as e:
        resultado_label.config(text="Error al entrenar el modelo.", fg="#b71c1c")
        messagebox.showerror(
            "Error",
            f"Ocurrio un problema al entrenar el modelo:\n{e}"
        )


def evaluar_modelo():
    try:
        accuracy = evaluate_model()
        resultado_label.config(
            text=f"Accuracy del modelo: {accuracy:.2f}%",
            fg="#0d47a1"
        )
        messagebox.showinfo(
            "Evaluacion completada",
            f"Precision del modelo: {accuracy:.2f}%"
        )

    except Exception as e:
        resultado_label.config(text="Error al evaluar el modelo.", fg="#b71c1c")
        messagebox.showerror(
            "Error",
            f"Ocurrio un problema al evaluar el modelo:\n{e}"
        )


def mostrar_graficas():
    try:
        show_model_graphs()
        resultado_label.config(
            text="Graficas generadas correctamente",
            fg="#1b5e20"
        )

    except Exception as e:
        resultado_label.config(text="Error al generar las graficas.", fg="#b71c1c")
        messagebox.showerror(
            "Error",
            f"Ocurrio un problema al generar las graficas:\n{e}"
        )


ventana = tk.Tk()
ventana.title("Clasificador de Diagnostico")
ventana.geometry("560x420")
ventana.resizable(False, False)
ventana.configure(bg="#f3f4f6")

titulo = tk.Label(
    ventana,
    text="Clasificador de Diagnostico",
    font=("Helvetica", 18, "bold"),
    bg="#f3f4f6",
    fg="#1c2733"
)
titulo.pack(pady=(20, 10))

subtitulo = tk.Label(
    ventana,
    text="Entrena, evalua y visualiza el comportamiento del modelo.",
    font=("Helvetica", 11),
    bg="#f3f4f6",
    fg="#4a4a4a"
)
subtitulo.pack(pady=(0, 20))

botones_frame = tk.Frame(ventana, bg="#f3f4f6")
botones_frame.pack()

btn_entrenar = tk.Button(
    botones_frame,
    text="Entrenar Modelo",
    font=("Helvetica", 12, "bold"),
    bg="#1976d2",
    fg="white",
    activebackground="#1565c0",
    activeforeground="white",
    width=18,
    height=2,
    relief="flat",
    command=entrenar_modelo
)
btn_entrenar.grid(row=0, column=0, padx=10, pady=8)

btn_evaluar = tk.Button(
    botones_frame,
    text="Evaluar Precision",
    font=("Helvetica", 12, "bold"),
    bg="#388e3c",
    fg="white",
    activebackground="#2e7d32",
    activeforeground="white",
    width=18,
    height=2,
    relief="flat",
    command=evaluar_modelo
)
btn_evaluar.grid(row=0, column=1, padx=10, pady=8)

btn_graficas = tk.Button(
    botones_frame,
    text="Mostrar Graficas",
    font=("Helvetica", 12, "bold"),
    bg="#f57c00",
    fg="white",
    activebackground="#ef6c00",
    activeforeground="white",
    width=38,
    height=2,
    relief="flat",
    command=mostrar_graficas
)
btn_graficas.grid(row=1, column=0, columnspan=2, padx=10, pady=8)

resultado_label = tk.Label(
    ventana,
    text="Esperando accion...",
    font=("Helvetica", 12),
    bg="#eef3f8",
    fg="#0d47a1",
    width=45,
    height=2,
    relief="ridge",
    bd=2
)
resultado_label.pack(pady=(20, 0))
