import time
import os
import csv
import numpy as np
import ray
import psutil
import cupy as cp
import pynvml
from codecarbon import EmissionsTracker

LEN_ALPHABET = 26
REPETITION = 5
NUM_CORES = 4

# --- Clase para rastrear hardware (CPU, RAM, GPU, VRAM) y energía ---
class ResourceTracker:
    def __init__(self):
        self.emissions_tracker = EmissionsTracker(log_level='ERROR', save_to_file=False)
        self.process = psutil.Process(os.getpid())
        
        # Inicializar NVML para métricas de GPU
        try:
            pynvml.nvmlInit()
            self.nvml_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Tomamos la GPU 0
        except Exception:
            self.nvml_available = False
        
    def __enter__(self):
        self.emissions_tracker.start()
        self.process.cpu_percent(interval=None) 
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.emissions_tracker.stop()
        self.end_energy = self.emissions_tracker.final_emissions_data.energy_consumed
        
        # Captura de métricas finales
        self.end_cpu = self.get_cluster_cpu()
        self.end_mem = self.get_cluster_memory()
        
        if self.nvml_available:
            try:
                # Obtener uso de GPU y memoria usada
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.end_gpu = util.gpu
                self.end_vram = mem_info.used / (1024 * 1024)
            except Exception:
                self.end_gpu, self.end_vram = 0.0, 0.0
        else:
            self.end_gpu, self.end_vram = 0.0, 0.0

        if self.nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def get_cluster_memory(self):
        total_mem = self.process.memory_info().rss
        for child in self.process.children(recursive=True):
            try:
                total_mem += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_mem / (1024 * 1024)

    def get_cluster_cpu(self):
        total_cpu = self.process.cpu_percent(interval=None)
        for child in self.process.children(recursive=True):
            try:
                total_cpu += child.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_cpu / NUM_CORES

    def get_metrics_dict(self, name, t_total, t_avg):
        return {
            "Ejecucion": name,
            "Tiempo_Total_s": round(t_total, 4),
            "Tiempo_Promedio_s": round(t_avg, 4),
            "Energia_mWh": round(self.end_energy * 1000000, 6),
            "Uso_CPU_Porcentaje": round(self.end_cpu, 1),
            "RAM_MB": round(self.end_mem, 2),
            "Uso_GPU_Porcentaje": round(self.end_gpu, 1),
            "VRAM_MB": round(self.end_vram, 2)
        }

    def print_metrics(self, metrics):
        print(f"[{metrics['Ejecucion']}]:")
        print(f"    Tiempo Total:       {metrics['Tiempo_Total_s']} s | Promedio: {metrics['Tiempo_Promedio_s']} s")
        print(f"    Energía consumida:  {metrics['Energia_mWh']} mWh")
        print(f"    Uso Estimado CPU:   {metrics['Uso_CPU_Porcentaje']} %")
        print(f"    Memoria RAM:        {metrics['RAM_MB']} MB")
        print(f"    Uso GPU:            {metrics['Uso_GPU_Porcentaje']} %")
        print(f"    Memoria VRAM:       {metrics['VRAM_MB']} MB")
        print("-" * 50)


# --- Funciones de utilidad ---
def toLowerCase(text):
    return text.lower()

def textToNum(text):
    base = ord('a')
    return [ord(char) - base if char != ' ' else -65 for char in text]

def avgTime(t_list):
    return sum(t_list) / len(t_list) if t_list else 0


# --- 1. Vigenère Puro ---
def CDNormal(text, key, rep):
    total_time = []
    for _ in range(rep):
        start = time.time()
        cypher_array = []
        key_count = 0
        for j in range(len(text)):
            if text[j] == -65:
                cypher_array.append(-65)
            else:
                cypher_array.append((text[j] + key[key_count]) % LEN_ALPHABET)
                key_count = (key_count + 1) % len(key)
                
        decypher_array = []
        key_count = 0
        for j in range(len(cypher_array)):
            if cypher_array[j] == -65:
                decypher_array.append(-65)
            else:
                decypher_array.append((cypher_array[j] - key[key_count]) % LEN_ALPHABET)
                key_count = (key_count + 1) % len(key)
        total_time.append(time.time() - start)
    return sum(total_time), avgTime(total_time)


# --- 2. Vigenère con Ray Puro + CPU ---
@ray.remote(num_cpus=4)
def cypher_chunk_ray_CPU(chunk, start_key_idx, key):
    cypher_text = []
    key_count = start_key_idx
    for char in chunk:
        if char == -65:
            cypher_text.append(-65)
        else:
            cypher_text.append((char + key[key_count]) % LEN_ALPHABET)
            key_count = (key_count + 1) % len(key)
    return cypher_text, key_count

@ray.remote(num_cpus=4)
def decypher_chunk_ray_CPU(chunk, start_key_idx, key):
    decypher_text = []
    key_count = start_key_idx
    for char in chunk:
        if char == -65:
            decypher_text.append(-65)
        else:
            decypher_text.append((char - key[key_count]) % LEN_ALPHABET)
            key_count = (key_count + 1) % len(key)
    return decypher_text, key_count

def CDRaySolo_CPU(text, key, rep):
    chunk_size = (len(text) + NUM_CORES - 1) // NUM_CORES
    valid_counts = []
    current_valid = 0
    for char in text:
        valid_counts.append(current_valid)
        if char != -65:
            current_valid += 1
            
    chunks_indices = range(0, len(text), chunk_size)
    start_key_indices = [valid_counts[idx] % len(key) for idx in chunks_indices]
    text_chunks = [text[idx:idx + chunk_size] for idx in chunks_indices]
    
    total_time = []
    for _ in range(rep):
        start = time.time()
        
        cypher_futures = [
            cypher_chunk_ray_CPU.remote(text_chunks[i], start_key_indices[i], key)
            for i in range(len(text_chunks))
        ]
        cypher_results = ray.get(cypher_futures)
        cypher_array = [char for res in cypher_results for char in res[0]]
        
        cypher_chunks = [cypher_array[idx:idx + chunk_size] for idx in chunks_indices]
        decypher_futures = [
            decypher_chunk_ray_CPU.remote(cypher_chunks[i], start_key_indices[i], key)
            for i in range(len(cypher_chunks))
        ]
        decypher_results = ray.get(decypher_futures)
        
        total_time.append(time.time() - start)
    return sum(total_time), avgTime(total_time)


# --- 3. Vigenère con Ray + NumPy + CPU ---
@ray.remote(num_cpus=4)
def process_chunk_numpy_CPU(text_np_chunk, aligned_key_chunk, mask_chunk, mode="cypher"):
    result = text_np_chunk.copy()
    if mode == "cypher":
        result[mask_chunk] = (text_np_chunk[mask_chunk] + aligned_key_chunk[mask_chunk]) % LEN_ALPHABET
    else:
        result[mask_chunk] = (text_np_chunk[mask_chunk] - aligned_key_chunk[mask_chunk]) % LEN_ALPHABET
    return result

def CDRayNumpy_CPU(text, key, rep):
    text_np = np.array(text)
    key_np = np.array(key)
    mask = (text_np != -65)
    
    valid_indices = np.cumsum(mask) - 1
    valid_indices = np.where(mask, valid_indices, 0)
    aligned_key = key_np[valid_indices % len(key_np)]
    
    chunk_size = (len(text_np) + NUM_CORES - 1) // NUM_CORES
    
    text_chunks = [text_np[i:i + chunk_size] for i in range(0, len(text_np), chunk_size)]
    key_chunks = [aligned_key[i:i + chunk_size] for i in range(0, len(aligned_key), chunk_size)]
    mask_chunks = [mask[i:i + chunk_size] for i in range(0, len(mask), chunk_size)]
    
    total_time = []
    for _ in range(rep):
        start = time.time()
        
        cypher_futures = [
            process_chunk_numpy_CPU.remote(text_chunks[i], key_chunks[i], mask_chunks[i], "cypher")
            for i in range(len(text_chunks))
        ]
        cypher_res_chunks = ray.get(cypher_futures)
        cypher_np = np.concatenate(cypher_res_chunks)
        
        cypher_chunks_split = [cypher_np[i:i + chunk_size] for i in range(0, len(cypher_np), chunk_size)]
        decypher_futures = [
            process_chunk_numpy_CPU.remote(cypher_chunks_split[i], key_chunks[i], mask_chunks[i], "decypher")
            for i in range(len(cypher_chunks_split))
        ]
        _ = ray.get(decypher_futures)
        
        total_time.append(time.time() - start)
    return sum(total_time), avgTime(total_time)


# --- 4.Vigenère con Ray + CuPy + GPU ---
@ray.remote(num_gpus=1)
def process_chunk_cupy_GPU(text_np, aligned_key, mask, mode="cypher"):
    # Enviamos el arreglo completo directamente a la GPU
    text_cp = cp.asarray(text_np)
    key_cp = cp.asarray(aligned_key)
    mask_cp = cp.asarray(mask)
    
    result = text_cp.copy()
    
    if mode == "cypher":
        result[mask_cp] = (text_cp[mask_cp] + key_cp[mask_cp]) % LEN_ALPHABET
    else:
        result[mask_cp] = (text_cp[mask_cp] - key_cp[mask_cp]) % LEN_ALPHABET
        
    # Devolver el arreglo completo a la memoria CPU
    return cp.asnumpy(result)


def CDRayCupy_GPU(text, key, rep):
    # 1. Preparación de los arrays completos en NumPy (CPU)
    text_np = np.array(text, dtype=np.int32)
    key_np = np.array(key, dtype=np.int32)
    mask = (text_np != -65)
    
    valid_indices = np.cumsum(mask) - 1
    valid_indices = np.where(mask, valid_indices, 0)
    aligned_key = key_np[valid_indices % len(key_np)]
    
    total_time = []
    for _ in range(rep):
        start = time.time()
        
        # Cifrado: Se envía TODO el bloque a un único Worker con GPU
        cypher_future = process_chunk_cupy_GPU.remote(text_np, aligned_key, mask, "cypher")
        cypher_np = ray.get(cypher_future)
        
        # Descifrado: Se envía el resultado completo de vuelta a la GPU
        decypher_future = process_chunk_cupy_GPU.remote(cypher_np, aligned_key, mask, "decypher")
        _ = ray.get(decypher_future)
        
        total_time.append(time.time() - start)
        
    return sum(total_time), avgTime(total_time)


# --- Bloque Principal ---
if __name__ == "__main__":
    
    if os.path.exists("text.txt"):
        with open("text.txt", "r") as f:
            raw_text = f.read()
    else:
        raw_text = "high performance computing " * 80000 

    num_text = textToNum(toLowerCase(raw_text))
    num_key = textToNum(toLowerCase("High Performance Computing"))

    print("\n" + "="*70)
    print("EJECUTANDO BENCHMARK INTEGRAL CON CUPY Y MÉTRICAS")
    print("="*70 + "\n")

    all_metrics = []

    # 1. Python Puro
    with ResourceTracker() as tracker:
        t_total, t_avg = CDNormal(num_text, num_key, REPETITION)
    m = tracker.get_metrics_dict("PYTHON PURO", t_total, t_avg)
    tracker.print_metrics(m)
    all_metrics.append(m)

    # Inicializar Ray
    ray.init(num_cpus=NUM_CORES, num_gpus=1, logging_level=50)

    # 2. Ray Solo CPU
    with ResourceTracker() as tracker:
        t_total, t_avg = CDRaySolo_CPU(num_text, num_key, REPETITION)
    m = tracker.get_metrics_dict("RAY SOLO CPU", t_total, t_avg)
    tracker.print_metrics(m)
    all_metrics.append(m)

    # 3. Ray + NumPy CPU
    with ResourceTracker() as tracker:
        t_total, t_avg = CDRayNumpy_CPU(num_text, num_key, REPETITION)
    m = tracker.get_metrics_dict("RAY + NUMPY CPU", t_total, t_avg)
    tracker.print_metrics(m)
    all_metrics.append(m)

    # 4. Ray + CuPy GPU
    with ResourceTracker() as tracker:
        t_total, t_avg = CDRayCupy_GPU(num_text, num_key, REPETITION)
    m = tracker.get_metrics_dict("RAY + CUPY GPU", t_total, t_avg)
    tracker.print_metrics(m)
    all_metrics.append(m)
    
    # --- Guardar a CSV ---
    csv_path = "/app/output/metrics.csv"
    keys = all_metrics[0].keys()
    with open(csv_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_metrics)
        
    print(f"\n[INFO] Métricas guardadas exitosamente en: {csv_path}")
    print("=" * 70)
    ray.shutdown()