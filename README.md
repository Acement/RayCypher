# RayCypher

Este es un programa que usa el cifrado Vigenère para comparar el tiempo de cifrado y descifrado de diferentes herramientas de HPC, y al finalizar se generan gráficos para facilitar la comparación.

### Requisitos:

- **Docker**

### Como se usa:
1. Clonar el repositorio.
2. Abrir la terminal en la carpeta del repositorio clonado.
3. Ejecutar los siguientes comandos:

``` sh
docker build --network=host -t vigenere-benchmark .
docker run --rm --gpus all -v "$(pwd)/output:/app/output" vigenere-benchmark
```
### Librerías
Se usaron las siguientes librerías:
- Ray = 2.35.0
- NumPy = 1.26.4
- Cupy-cuda12x
- Codecarbon = 2.4.2
- Psutil = 5.9.8
- Pynvml = 11.5.0
- Matplotlib = 3.8.4
- Pandas = 2.2.2
