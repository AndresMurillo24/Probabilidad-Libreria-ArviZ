# Ejemplo: simulación de aciertos en Baloto usando ArviZ
# Cada línea tiene un comentario que explica su función.

import numpy as np                        # importa NumPy para manejo de arreglos y aleatoriedad
import arviz as az                        # importa ArviZ para análisis y resumen de resultados
import matplotlib.pyplot as plt           # importa matplotlib para dibujar (si queremos gráficos)

np.random.seed(42)                        # fija la semilla aleatoria para resultados reproducibles

total_numeros = 45                        # número total de bolas disponibles en el Baloto (1..45)
numeros_elegidos = 6                      # cuántos números se eligen por boleto (6)
n_simulaciones = 50000                    # cuántos boletos aleatorios vamos a simular

# generamos un "resultado oficial" del sorteo: 6 números distintos elegidos sin reemplazo
resultado_oficial = np.random.choice(
    range(1, total_numeros + 1),          # valores posibles (1..45)
    size=numeros_elegidos,                # tomar 6 números
    replace=False                         # sin reemplazo: no se repiten números
)

# inicializamos una lista Python para guardar el número de coincidencias por boleto simulado
aciertos = []                             # lista vacía donde almacenaremos los conteos de aciertos

# bucle que genera n_simulaciones boletos y cuenta coincidencias con el resultado oficial
for _ in range(n_simulaciones):           # repetir la simulación n_simulaciones veces
    boleto = np.random.choice(            # generar un boleto aleatorio
        range(1, total_numeros + 1),      # valores posibles (1..45)
        size=numeros_elegidos,            # tomar 6 números por boleto
        replace=False                     # sin reemplazo
    )
    coincidencias = len(set(boleto) & set(resultado_oficial))
    # calcula la intersección entre boleto y resultado_oficial y obtiene su tamaño
    aciertos.append(coincidencias)        # añade el número de coincidencias a la lista 'aciertos'

# convertimos la lista de aciertos a un arreglo de NumPy (forma requerida por muchos análisis)
aciertos_array = np.array(aciertos)       # convierte la lista a numpy array (shape: (n_simulaciones,))

# creamos un objeto InferenceData con ArviZ a partir del arreglo de aciertos
# az.from_dict convierte diccionarios de arrays a un InferenceData (formato estándar de ArviZ)
idata = az.from_dict(posterior={"aciertos": aciertos_array})

# imprimimos un encabezado para la salida por consola (para que quede claro)
print("\n=== Resumen de aciertos en Baloto (simulación) ===")

# az.summary calcula y muestra estadísticos básicos: mean, sd, y intervalos de credibilidad (HDI)
# aquí pedimos el resumen solo para la variable "aciertos"
summary = az.summary(idata, var_names=["aciertos"])  # obtiene un DataFrame con los estadísticos
print(summary)                                       # muestra el resumen por consola

# (opcional) mostramos una distribución posterior simple con az.plot_posterior
# az.plot_posterior: dibuja la "distribución posterior" (aquí, la frecuencia de aciertos)
az.plot_posterior(idata, var_names=["aciertos"], hdi_prob=0.95)
plt.suptitle("Distribución de aciertos en Baloto (simulación)")
plt.show()                                          # abre la ventana del gráfico (si se desea)

# (opcional) traza de las simulaciones con az.plot_trace
# az.plot_trace dibuja un traceplot (serie temporal de muestras) y un histograma por variable
az.plot_trace(idata, var_names=["aciertos"])
plt.suptitle("Trace plot - aciertos simulados")
plt.show()

# (opcional) Posterior Predictive Check (PPC) simple:
# construimos un "posterior_predictive" a partir de las mismas simulaciones para usar az.plot_ppc
ppc_idata = az.from_dict(posterior_predictive={"aciertos": aciertos_array})
# az.plot_ppc compara datos observados con muestras predictivas para verificar ajuste
az.plot_ppc(ppc_idata, num_pp_samples=100)
plt.suptitle("PPC - comparación predictiva (simulación)")
plt.show()

