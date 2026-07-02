import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker

def generate_plots():
    plt.rcParams.update({'font.size': 13})
    csv_path = "/app/output/metrics.csv"
    output_dir = "/app/output"
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontró el archivo de métricas en {csv_path}")
        return

    # Cargar datos
    df = pd.read_csv(csv_path)
    
    # Colores consistentes para cada tecnología
    colors = {
        "PYTHON PURO": "#d9534f",
        "RAY SOLO CPU": "#f0ad4e",
        "RAY + NUMPY CPU": "#5bc0de",
        "RAY + CUPY GPU": "#5cb85c"
    }
    
    # --- GRÁFICO 1A: Vista General (Todas las Técnicas) ---
    plt.figure(figsize=(10, 6))
    for name, group in df.groupby("Ejecucion"):
        # Corregido de 'ax.plot' a 'plt.plot' al no usar subplots
        plt.plot(group["Iteracion"], group["Tiempo_s"], marker='o', markersize=4, 
                 label=name, color=colors.get(name, "#777777"), linewidth=1.5)
        
    plt.title("Vista General de Tiempos de Ejecución (Todas las Técnicas)", fontsize=13, fontweight='bold')
    plt.xlabel("Iteración / Repetición", fontsize=10)
    plt.ylabel("Tiempo (segundos)", fontsize=10)
    plt.yscale("log")
    
    # Formateador para mostrar 0.0001s en vez de 10^-4
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    
    plot1a_path = os.path.join(output_dir, "tiempo_vista_general.png")
    plt.savefig(plot1a_path, dpi=150)
    plt.close()
    print(f"[INFO] Gráfico de vista general guardado en: {plot1a_path}")


    # --- GRÁFICO 1B: Vista Zoom (Solo Frameworks de Alto Rendimiento) ---
    df_fast = df[df["Ejecucion"] != "PYTHON PURO"]
    
    plt.figure(figsize=(10, 6))
    if not df_fast.empty:
        for name, group in df_fast.groupby("Ejecucion"):
            plt.plot(group["Iteracion"], group["Tiempo_s"], marker='s', markersize=4, 
                     label=name, color=colors.get(name), linewidth=1.5)
            
        plt.title("Herramientas de Alto Rendimiento (Escala Lineal)", fontsize=13, fontweight='bold')
        plt.xlabel("Iteración / Repetición", fontsize=10)
        plt.ylabel("Tiempo (segundos)", fontsize=10)
        
        # Forzar formato decimal plano en el eje Y
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper right", fontsize=9)
    else:
        plt.text(0.5, 0.5, "No hay datos de rendimiento", ha='center', va='center')
        
    plt.tight_layout()
    plot1b_path = os.path.join(output_dir, "tiempo_alto_rendimiento.png")
    plt.savefig(plot1b_path, dpi=150)
    plt.close()
    print(f"[INFO] Gráfico de alto rendimiento guardado en: {plot1b_path}")


    # --- GRÁFICO 2: Comparativa del Tiempo Promedio (Barras) ---
    plt.figure(figsize=(10, 6))
    avg_times = df.groupby("Ejecucion")["Tiempo_s"].mean().sort_values(ascending=False)
    
    # Mapear colores según el orden del groupby index
    bar_colors = [colors.get(name, "#777777") for name in avg_times.index]
    bars = plt.bar(avg_times.index, avg_times.values, color=bar_colors, edgecolor='black', alpha=0.8)
    plt.title("Comparativa de Tiempo Promedio (Escala Logarítmica)", fontsize=14, fontweight='bold')
    plt.ylabel("Tiempo Promedio (segundos)", fontsize=12)
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    plt.grid(True, which="both", linestyle="--", alpha=0.3, axis='y')
    
    # Añadir etiquetas con formato dinámico de decimales sobre las barras
    for bar in bars:
        height = bar.get_height()
        fmt = f'{height:.6f}s' if height < 0.01 else f'{height:.4f}s'
        plt.text(bar.get_x() + bar.get_width()/2.0, height, fmt, 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "tiempo_promedio_comparativo.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"[INFO] Gráfico de barras comparativo guardado en: {plot2_path}")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERANDO GRÁFICOS OPTIMIZADOS EN SEGUNDOS")
    print("="*70)
    generate_plots()