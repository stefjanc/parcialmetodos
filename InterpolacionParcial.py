from typing import Callable
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def calc_coeficientes(x, y):
    n = len(x)
    c = y.copy()
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            c[j] = (c[j] - c[j - 1]) / (x[j] - x[j - i])
    return c

def calc_difdev(x,y):
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x[i + j] - x[i])
    return F[0, :]

def interpolador(x, coeficientes):
    x_sym = sp.symbols('x')
    n = len(coeficientes)
    p = coeficientes[n - 1]
    for i in range(n - 2, -1, -1):
        p = p * (x_sym - x[i]) + coeficientes[i]
    return p

def symPolin(x, y, polin):
    polinomio_expandido = sp.expand(polin)
    polinomio_expandido = polinomio_expandido.subs('t', 'x')
    polinomio_numerico = sp.lambdify('x', polinomio_expandido)
    return polinomio_numerico, polinomio_expandido
    
def graficar(x, y, polin, titulo):
    x_vals = np.linspace(min(x), max(x), 100)
    y_vals = polin(x_vals)
    plt.scatter(x, y, color='red', label='Puntos')
    plt.plot(x_vals, y_vals, color='blue', label=titulo)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titulo)
    plt.legend()
    plt.show()
    
def lagrange(x, y):
    n = len(x)
    t = sp.Symbol('t')
    polinomio = 0

    for i in range(n):
        termino = y[i]
        for j in range(n):
            if j != i:
                termino *= (t - x[j]) / (x[i] - x[j])
        polinomio += termino
    return polinomio

def newton_secante(f: Callable[[float], float], df, a: float, b: float, epsilon: float, maxiter: int) -> float:
    rootFound = False
    x = a  # Valor inicial de X
    fx = f(x)   
    iteracion = 0
    
    while abs(fx) > epsilon and iteracion < maxiter:
        # Newton
        print(f"Realizamos una iteración de Newton en intervalo [{a:.6f}, {b:.6f}]")
        try:
            dx = -fx / df(x)
        except ZeroDivisionError:
            print("Se intentó dividir por 0. Cambie los parámetros iniciales e intente de nuevo.")
            return 0
        x += dx
        fx = f(x)
        iteracion += 1
        print(f"Iteracion de Newton {iteracion}: x = {x:.6f}, f(x) = {fx:.6f}\n")
        
        if abs(fx) <= epsilon:
            rootFound = True
            break
        
        # Secante
        print(f"Realizamos una iteración de Secante con x={x:.6f}")
        try:
            dx = -fx * (b - x) / (fx - f(b))
        except ZeroDivisionError:
            print("Se intentó dividir por 0. Cambie los parámetros iniciales e intente de nuevo.")
            return 0
        a, b = b, x
        x += dx
        fx = f(x)
        print(f"Iteracion Secante {iteracion}: x = {x:.6f}, f(x) = {fx:.6f}\n")

        
        if abs(fx) <= epsilon:
            rootFound = True
            break
    if rootFound:
        print(f"Se logró aproximar hacia una raiz en el punto : {x:.6f}")
    else:
        print(f"No se encontró raíz en {maxiter} iteraciones.")
        
def raiz_polinomio(p_expr, a, b, epsilon, max_iter):
    x = sp.Symbol('x')
    p_deriv_expr = sp.diff(p_expr, x)
    p = sp.lambdify(x, p_expr)
    p_deriv = sp.lambdify(x, p_deriv_expr)
    f = lambda x_val: p(x_val)
    df = lambda x_val: p_deriv(x_val)
    raiz = newton_secante(f, df, a, b, epsilon, max_iter)
    

'''
x = np.random.uniform(-30, 30, size=20)
y = np.random.uniform(-30, 30, size=20)
'''


x = np.array([1, 0, -3])
y = np.array([2, 4, -2])

print('Puntos dados:\n')
for i in range(len(x)):
    print(f'({x[i]}, {y[i]})\n')
coeficientes = calc_coeficientes(x,y)
polinomioC = interpolador(x, coeficientes)
polinomio_n, polinomio_e = symPolin(x, y, polinomioC)
print(f"Polinomio Interpolador:\n{polinomio_e}")
graficar(x, y, polinomio_n, "Polinomio Interpolador")