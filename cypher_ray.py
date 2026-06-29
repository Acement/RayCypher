import time
import os
import numpy as np
import ray
import psutil
from codecarbon import EmissionsTracker

# Intentar importar la librería de NVIDIA para la GPU
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

LEN_ALPHABET = 26
REPETITION = 20
NUM_CORES = 4

# --- Clase para rastrear hardware y energía ---
class ResourceTracker:
    def __init__(self):
        self.emissions_tracker = EmissionsTracker(log_level='ERROR', save_to_file=False)
        self.process = psutil.Process(os.getpid())
        
    def __enter__(self):
        self.emissions_tracker.start()
        # Captura inicial de CPU (porcentaje) y memoria
        self.process.cpu_percent(interval=None) 
        self.start_mem = self.get_cluster_memory()
        self.start_gpu = self.get_gpu_usage() if GPU_AVAILABLE else 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.emissions_tracker.stop()
        self.end_energy = self.emissions_tracker.final_emissions_data.energy_consumed
        
        # Métrica promedio/pico al finalizar
        self.end_cpu = self.get_cluster_cpu()
        self.end_mem = self.get_cluster_memory()
        self.end_gpu = self.get_gpu_usage() if GPU_AVAILABLE else 0

    def get_cluster_memory(self):
        # Mide la memoria del proceso padre + todos los workers hijos creados por Ray/Pool
        total_mem = self.process.memory_info().rss
        for child in self.process.children(recursive=True):
            try:
                total_mem += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_mem / (1024 * 1024) # Convertir a MB

    def get_cluster_cpu(self):
        # Porcentaje de CPU usado por todo el árbol de procesos
        total_cpu = self.process.cpu_percent(interval=None)
        for child in self.process.children(recursive=True):
            try:
                total_cpu += child.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_cpu / NUM_CORES # Normalizado por hilos asignados

    def get_gpu_usage(self):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            res = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return res.gpu # Porcentaje de uso de la GPU
        except Exception:
            return 0

    def print_metrics(self, name, t_total, t_avg):
        print(f"[{name}]:")
        print(f"    Tiempo Total:       {t_total:.4f} s | Promedio: {t_avg:.4f} s")
        print(f"    Energía consumida:  {self.end_energy:.6f} kWh")
        print(f"    Uso Estimado CPU:   {self.end_cpu:.1f} %")
        print(f"    Memoria RAM RAM:    {self.end_mem:.2f} MB")
        if GPU_AVAILABLE:
            print(f"    Uso de GPU (NVIDIA): {self.end_gpu} %")
        else:
            print(f"    Uso de GPU:         N/A (No CUDA/NVIDIA detectada)")
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

# --- 2. Vigenère con Ray Puro ---
@ray.remote
def cypher_chunk_ray(chunk, start_key_idx, key):
    cypher_text = []
    key_count = start_key_idx
    for char in chunk:
        if char == -65:
            cypher_text.append(-65)
        else:
            cypher_text.append((char + key[key_count]) % LEN_ALPHABET)
            key_count = (key_count + 1) % len(key)
    return cypher_text, key_count

@ray.remote
def decypher_chunk_ray(chunk, start_key_idx, key):
    decypher_text = []
    key_count = start_key_idx
    for char in chunk:
        if char == -65:
            decypher_text.append(-65)
        else:
            decypher_text.append((char - key[key_count]) % LEN_ALPHABET)
            key_count = (key_count + 1) % len(key)
    return decypher_text, key_count

def CDRaySolo(text, key, rep):
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
            cypher_chunk_ray.remote(text_chunks[i], start_key_indices[i], key)
            for i in range(len(text_chunks))
        ]
        cypher_results = ray.get(cypher_futures)
        cypher_array = [char for res in cypher_results for char in res[0]]
        
        cypher_chunks = [cypher_array[idx:idx + chunk_size] for idx in chunks_indices]
        decypher_futures = [
            decypher_chunk_ray.remote(cypher_chunks[i], start_key_indices[i], key)
            for i in range(len(cypher_chunks))
        ]
        decypher_results = ray.get(decypher_futures)
        
        total_time.append(time.time() - start)
    return sum(total_time), avgTime(total_time)

# --- 3. Vigenère con Ray + NumPy ---
@ray.remote
def process_chunk_numpy(text_np_chunk, aligned_key_chunk, mask_chunk, mode="cypher"):
    result = text_np_chunk.copy()
    if mode == "cypher":
        result[mask_chunk] = (text_np_chunk[mask_chunk] + aligned_key_chunk[mask_chunk]) % LEN_ALPHABET
    else:
        result[mask_chunk] = (text_np_chunk[mask_chunk] - aligned_key_chunk[mask_chunk]) % LEN_ALPHABET
    return result

def CDRayNumpy(text, key, rep):
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
            process_chunk_numpy.remote(text_chunks[i], key_chunks[i], mask_chunks[i], "cypher")
            for i in range(len(text_chunks))
        ]
        cypher_res_chunks = ray.get(cypher_futures)
        cypher_np = np.concatenate(cypher_res_chunks)
        
        cypher_chunks_split = [cypher_np[i:i + chunk_size] for i in range(0, len(cypher_np), chunk_size)]
        decypher_futures = [
            process_chunk_numpy.remote(cypher_chunks_split[i], key_chunks[i], mask_chunks[i], "decypher")
            for i in range(len(cypher_chunks_split))
        ]
        _ = ray.get(decypher_futures)
        
        total_time.append(time.time() - start)
    return sum(total_time), avgTime(total_time)

# --- Bloque Principal ---
if __name__ == "__main__":
    ray.init(num_cpus=NUM_CORES, logging_level=50)
    
    if os.path.exists("text.txt"):
        with open("text.txt", "r") as f:
            raw_text = f.read()
#    else:
#        raw_text = "high performance computing " * 80000 

    num_text = textToNum(toLowerCase(raw_text))
    num_key = textToNum(toLowerCase("High Performance Computing"))

    print("\n" + "="*70)
    print("EJECUTANDO BENCHMARK INTEGRAL (HARDWARE, TIEMPO Y ENERGÍA)")
    print("="*70 + "\n")

    # 1. Python Puro
    with ResourceTracker() as tracker:
        t_total, t_avg = CDNormal(num_text, num_key, REPETITION)
    tracker.print_metrics("PYTHON PURO", t_total, t_avg)

    # 2. Ray Solo
    with ResourceTracker() as tracker:
        t_total, t_avg = CDRaySolo(num_text, num_key, REPETITION)
    tracker.print_metrics("RAY SOLO", t_total, t_avg)

    # 3. Ray + NumPy
    with ResourceTracker() as tracker:
        t_total, t_avg = CDRayNumpy(num_text, num_key, REPETITION)
    tracker.print_metrics("RAY + NUMPY", t_total, t_avg)
    
    print("=" * 70)
    ray.shutdown()
    if GPU_AVAILABLE:
        pynvml.nvmlShutdown()