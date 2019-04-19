from matrix import *

def dot_product(A, B):
    result = 'matrix'
    if A.height == B.height and A.width == B.width:
        result = 'scalar'
        A = A.flip()
    elif A.width != B.height:
        if A.height == B.height:
            A = A.flip()
        else:
            print('Width of A != Height of B. Exiting...')
            return None
    height = A.height
    width = B.width
    degree = B.height
    C = Matrix(height, width)
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            total = 0
            for k in range(1, degree + 1):
                total += A.get(i, k) * B.get(k, j)
            C.set(i, j, total)
    if result == 'scalar':
        return C.get(1, 1)
    elif result == 'matrix':
        return C

def equilibrium(B, x, cycles=1000):
    equilibrium_matrix = x.copy()
    for i in range(cycles):
        equilibrium_matrix = dot_product(B, equilibrium_matrix)
    return equilibrium_matrix

def print_equilibrium(B, x, cycles=1000):
    print('Initial matrices:')
    B.print_matrix()
    x.print_matrix()
    equilibrium_matrix = equilibrium(B, x, cycles)
    print('Equilibrium matrix:')
    equilibrium_matrix.print_matrix()
