from typing import Callable
import sympy as sp
from sympy.utilities.lambdify import lambdify
import sys

def enter():
    print('Presione cualquier tecla para continuar.')
    input()
    
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

x = sp.Symbol('x')

p_expr = input('Ingrese la función: ')      # **3 - 5*x**2 + 6*x - 2   3.54457826408169e-20*x**19 - 2.20222839623258e-19*x**18 - 1.14058217270898e-16*x**17 + 5.72838114627743e-16*x**16 + 1.54781332762955e-13*x**15 - 5.99810719057742e-13*x**14 - 1.15358538489949e-10*x**13 + 3.22249608603652e-10*x**12 + 5.15411589137241e-8*x**11 - 9.23272089898862e-8*x**10 - 1.41553870211711e-5*x**9 + 1.23292387275586e-5*x**8 + 0.00236119033363712*x**7 - 2.28216284176054e-5*x**6 - 0.228300209866605*x**5 - 0.195883432516188*x**4 + 11.5284123474663*x**3 + 20.8124240687849*x**2 - 231.887978817865*x - 652.696722386457
p_expr = sp.sympify(p_expr)
print('La función ingresada fue:\n', p_expr)
tolerancia = float(input('Ingrese la tolerancia/Epsilon (por ejemplo 1e-6):'))
iteraciones = int(input('Ingrese la cantidad máxima de iteraciones:'))
raices = sp.solve(p_expr)
raices_numerico = [r.evalf() for r in raices]
raices_complejas = any(sp.im(r) != 0 for r in raices_numerico)
if raices_complejas:
    print('Sympy no encontró raíces para esta función.')
    sys.exit()
print('Raíces en la función ingresada según Sympy: \n', raices)
enter()

p_deriv_expr = sp.diff(p_expr, x)

print('f(x) = ', p_expr)
print("\nf'(x) = ", p_deriv_expr,"\n")

# Convertimos las expresiones simbólicas en funciones numéricas
p = sp.lambdify(x, p_expr)
p_deriv = sp.lambdify(x, p_deriv_expr)

# Definimos los parámetros para el método de Newton-Raphson
a = 4
b = 6
epsilon = tolerancia
max_iter = iteraciones

# Definimos la función f(x) y su derivada df(x) para el método de Newton-Raphson
f = lambda x_val: p(x_val)
df = lambda x_val: p_deriv(x_val)

# Aplicamos el método de Newton-Raphson
root = newton_secante(f, df, a, b, epsilon, max_iter)