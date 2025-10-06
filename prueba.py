import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
print("Generar los datos")
x=np.linescape(-5, 5, 1000)
y=x**2+np.random.normal(0, 2, 1000)
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)
x_train=x_train.reshape(-1,1)
x_test=x_train.teshape(-1,1)
print(f"Datos de entrenamiento: {x_train.shape[0]} muestras")
print("Datos listos para la red")
print("\n Construyendo el modelo")
model=Sequential([
    Dense (32, activation='relu', input_shape=(1,)),
    Dense (32, activation='relu'),
    Dense(1)
])
model.sumary()
print("\n3. Copilando y entrenando el modelo")
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])
history=model.fit(x_train, y_train,
                  epochs=100,
                  verbose=0)
print("Entrenamiento finalizado")
print("\n4 Evaluando el modelo")
loss, mae=model.evaluate(x_test, y_test,verbose=0)
print(f"error cuadratico medio(loss/mse) en prueba:{loss:.2f}")
print(f"error absoluto medio (mae) en prueba:{mae:.2f}(el error promedio en la prediccion de y)")
x_range=np.linspace(x,min(),x.max(),100).reshape(-1,1)
predictions=model.predict(x_range)
print("\n3 visualizacion resultados")
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="datos originales (x vs x^2+nudo)", alpha=0.5, color='red')
plt.plot(x_range, predictions, color='blue',linewidth=3, label="curva de prediccion")
plt.title(f"regresion no lineal con real neuronal densa(mae:{mae:2.f})")
plt.xlabel('valor de salida(x)')
plt.ylabel('valor de salida (y)')
plt.legend()
plt.grid(True)
plt.show()