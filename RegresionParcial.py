import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

df = pd.read_excel()#'ACUMULADOS vs DIAS.xlsx')
x = df['día'].values
y = df['acumulados'].values

def duplicacion(y):
    tiempos_duplicacion = []
    y_duplicacion = []
    punto_partida = y[0]
    for i in range(len(y)):
        punto_duplicacion = punto_partida * 2
        index = np.where( y >= punto_duplicacion)[0]
        if len(index) > 0:
            indice_duplicacion = index[0]
        else:
            return tiempos_duplicacion, y_duplicacion
        tiempo_duplicacion = x[indice_duplicacion]
        tiempos_duplicacion.append(tiempo_duplicacion)
        punto_partida = y[indice_duplicacion]
        y_duplicacion.append(y[indice_duplicacion])
        
    return tiempos_duplicacion, y_duplicacion

# Funciones para los modelos
def f_linear(x, a, b):
    return a * x + b

def f_potencia(x, a, b):
    return b * np.power(x, a)

def f_exponencial(x, a, b):
    return b * np.exp(a * x)


# Ajuste de curvas y cálculo del coeficiente de correlación
popt_linear, _ = curve_fit(f_linear, x, y)
popt_potencia, _ = curve_fit(f_potencia, x, y)
popt_exponencial, _ = curve_fit(f_exponencial, x, y)

# Cálculo del coeficiente de correlación
r_linear = np.corrcoef(y, f_linear(x, *popt_linear))[0, 1]
r_potencia = np.corrcoef(y, f_potencia(x, *popt_potencia))[0, 1]
r_exponencial = np.corrcoef(y, f_exponencial(x, *popt_exponencial))[0, 1]

dy_dx_linear = np.gradient(f_linear(x, *popt_linear),x )
dy_dx_potencia = np.gradient(f_potencia(x, *popt_potencia), x)
d2y_dx2_linear = np.gradient(dy_dx_linear, x)
d2y_dx2_power = np.gradient(dy_dx_potencia, x)

tiempos_duplicacion, y_duplicacion = duplicacion(y)

#Curvas
plt.scatter(x, y, label='Datos', alpha=0.3)
plt.plot(x, f_linear(x, *popt_linear), label=f'Curva Lineal', color='green')
print(f'Curva Lineal:\nY = {popt_linear[1]}x^{popt_linear[0]}, r = {r_linear}')
plt.plot(x, f_potencia(x, *popt_potencia), label=f'Curva de Potencia', color='yellow')
print(f'Curva de Potencia:\nY = {popt_potencia[1]}x^{popt_potencia[0]}, r = {r_potencia}')
plt.plot(x, f_exponencial(x, *popt_exponencial), label=f'Curva Exponencial', color='blue')
print(f'Curva Exponencial:\nY = {popt_exponencial[1]}x^{popt_exponencial[0]}, r = {r_exponencial}')
plt.scatter(tiempos_duplicacion, y_duplicacion, color='red', label='Puntos de Duplicacion')
plt.ylim(-70000 , y.max())
plt.xlabel('Día')
plt.ylabel('Casos Acumulados')
plt.legend()
plt.show()

#Deriv 1
plt.plot(x, dy_dx_linear, label='Derivada Lineal', color = 'green')
plt.plot(x, dy_dx_potencia, label='Derivada de Potencia', color = 'yellow')
plt.xlabel('Día')
plt.ylabel('Primera Derivada')
plt.xlim(0, x.max())
plt.ylim(-1000, 15000)
plt.legend()
plt.show()

#Deriv 2
plt.plot(x, d2y_dx2_linear, label='Segunda Derivada Lineal')
plt.plot(x, d2y_dx2_power, label='Segunda Derivada Potencia')
plt.xlabel('Día')
plt.ylabel('Segunda Derivada')
plt.xlim(0, x.max())
plt.ylim(-150, 500)
plt.legend()
plt.show()