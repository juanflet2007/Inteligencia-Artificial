import numpy as np
def sigmoide (x):
    return 1/(1+np.exp(-x))
def derivada_sigmoide(x):
    return x*(1-x)
#preparacion de datos
datos_entrada=np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
salidas_deseadas=np.array([[0,1,1,0]]).T
np.random.seed(1)
pesos = 2 * np.random.random((2,1))-1
print(pesos)
epocas=60000
for i in range (epocas):
    z=np.dot(datos_entrada, pesos)
    capa_salida=sigmoide(z)
    error=salidas_deseadas-capa_salida
    if (1 % 10000)==0:
        print(f"Error en epoca {i}: {np.mean(np.abs(error)):.4f}")
        delta= error * derivada_sigmoide (capa_salida)
        ajustes_pesos = np.dot(datos_entrada.T,delta)
        pesos+= ajustes_pesos
print("Pesos aprendidos", pesos.round(4))
print("rediccion")
print(capa_salida.round(4))