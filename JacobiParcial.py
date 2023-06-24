import numpy as np
import matplotlib.pyplot as plt

def jacobi(A, b, maxiter, epsilon):
    haySol = False
    iteraciones = 0
    error = np.inf
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    L = np.tril(A, k=-1)  # Matriz triangular inferior
    U = np.triu(A, k=1)   # Matriz triangular superior
    D = np.diag(np.diag(A))        # Matriz diagonal
    errors = []

    if esCuadrado(A,b):    #and esDiagDom(A)
        Dinv = np.linalg.inv(D)
        while iteraciones < maxiter and error > epsilon:
            for i in range(n):
                x_new = (-Dinv) @ (L + U) @ x + Dinv @ b
                
            error = np.max(np.abs(x_new - x))        
            iteraciones+=1
            print(f"Iteracion {iteraciones}:\nx, y, z = {x_new}")
            print(f"Error de Iteracion {iteraciones}: {error}\n")
            x = np.around(x_new, decimals=6)
            errors.append(error)
            if(error < epsilon):
                haySol = True
                maxiter = iteraciones
                break            
        if(haySol):
            print("Solución: ", x_new)
            print("Iteraciones: ", iteraciones)
        else: print(f"No se encontró solución luego de {maxiter} iteraciones.")
        
        plt.plot(range(1, maxiter+1), errors)
        plt.xlabel('Número de iteración')
        plt.ylabel('Norma del error')
        plt.title('Convergencia del método de Jacobi')
        plt.show()
    
def esCuadrado(A, b):
    esCuadrada = len(A.shape) == 2 and A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0]
    print(f"esCuadrada: {esCuadrada}\n")
    if not(esCuadrada):
        print("El SEL ingresado no es cuadrado.")
        return False
    return True    
    
def esDiagDom(A):
    for i in range(A.shape[0]):
        esDiagDom = np.abs(A[i,i] >= (np.sum(np.abs(A[i, :]))) - np.abs(A[i, i]))
        if not esDiagDom:
            print("La matriz A no es dominantemente diagonal")
            return False
    print(f"esDiagDom: {esDiagDom}")
    return True

'''
A = np.array([[3, 2, -1],
              [2, 5, 1],
              [1, -1, 4]])
b = np.array([1, -4, 5])
'''
A = np.array([[4, 1, 1],
              [1, 4, 1],
              [1, 1, 4]])
b = np.array([5, 6, 7])


jacobi(A, b, 100, 1e-3)