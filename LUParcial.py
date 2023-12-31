import numpy as np
import sys
#np.seterr(all='raise')


def pivoteONo(A, n, L, U):
  
    print(f'La determinante de la matriz A es {np.linalg.det(A)}')    
    if not invertible(A):
        print('La matriz A ingresada no es invertible. Se intentará utilizar pivote:')
        pivot(A, n, L, U)   
    #else:
    for i in range(n):
            L[i, i] = 1.0       
    L, U = descomponer_lu(A, n, L, U)
    return L, U
        

def invertible(A):
    return np.linalg.det(A) != 0

def pivot(A, n, L, U):
    P = np.eye(n)
    for j in range(n):
        pivot_index = np.argmax(np.abs(A[j:, j])) + j
        P[[j, pivot_index], :] = P[[pivot_index, j], :]
        A[[j, pivot_index], :] = A[[pivot_index, j], :]
        L[j:, j] = A[j:, j] / A[j, j]
        U[j, j:] = A[j, j:]
        U[j+1:, j+1:] = A[j+1:, j+1:] - np.outer(L[j+1:, j], U[j, j+1:])
    return L, U    

def gen_lu(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    return n, L, U

def descomponer_lu(A, n, L, U):    
    print("Establecemos el estado inicial de la matriz L:")

    print(L, "\n")
    
    print("Y pasamos a calcular los elementos U y L:\n")
    for j in range(n):
        for i in range(j + 1):
            # Calcula la suma de los productos de elementos de U y L
            sum1 = sum(U[k, j] * L[i, k] for k in range(i))
            # Calcula el elemento U[i, j]
            U[i, j] = A[i, j] - sum1
            if np.isnan(U[j,j]):
                U[j,j] = 0            
        

        for i in range(j, n):
            # Calcula la suma de los productos de elementos de U y L
            sum2 = sum(U[k, j] * L[i, k] for k in range(j))
            # Calcula el elemento L[i, j]
            #try:
            L[i, j] = (A[i, j] - sum2) / U[j, j]
            if np.isnan(L[i,j]):
                L[i,j] = 1
            #except FloatingPointError:
            #    print("La matriz A no pudo hacerse Triangular.")
            #    sys.exit()
    print(f"U =\n {U}\n")
    print(f"L =\n {L}\n")                
    print(f'L x U =\n {L @ U}')
    return L, U

def lyb(U, L, x, y, b, n, cont):
    for i in range(n):
        sum1 = 0
        for k in range(i):
            sum1 += L[i,k] * y[k]
            cont+=2                   # +1 suma, +1 producto
        y[i] = b[i] - sum1
        cont+=1                       # +1 resta
    print(f"Y=\n{y}\n")
    cont = uxy(U, x, y, n, cont)
    return cont

def uxy(U, x, y, n, cont):
    for i in range(n-1, -1, -1):
        sum2 = 0
        for k in range(i+1, n):
            sum2 += U[i, k] * x[k]
            cont += 2            #+1 suma, +1 producto
        # Calcula el elemento x[i]
        x[i] = (y[i] - sum2) / U[i, i]
        cont+=2                     #+1 resta, +1 división
    print(f"X =\n{x}\n")
    return cont

def resolver_lu(A, b):
    cont = 0
    n, L, U = gen_lu(A)
    L, U = pivoteONo(A, n, L, U)

    y = np.zeros(n)
    x = np.zeros(n)
    
    cont = lyb(U, L, x, y, b, n, cont)
    print(f"Solución:\n X = {x[0]}\n Y = {x[1]}\n Z = {x[2]})\n La cantidad de operaciones elementales utilizadas fue: {cont}")

def puedoResolver(A, b):
    if len(A.shape) == 2 and A.shape[0] == A.shape[1] and A.shape[0] == b.shape[0]:
        resolver_lu(A, b)
    else:
        print("El SEL ingresado no es cuadrado. No sera posible implementar el metodo de descomposicion de LU.")
    
A = np.array([[2, 3, 1],
              [4, 7, 5],
              [6, 2, 3]])
b = np.array([1, 1, 6])


'''
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
b = np.array([4, 6, 8])
'''
puedoResolver(A, b)