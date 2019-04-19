from matrix import *
from matrix_operations import *

def main():
    B = Matrix(4, 4)
    B.build([
        [0.05, 0.45, 0.05, 0.05],
        [0.45, 0.05, 0.05, 0.85],
        [0.45, 0.45, 0.05, 0.05],
        [0.05, 0.05, 0.85, 0.05]
        ])
    x = Matrix(4, 1)
    x.set_column(1, [1000, 2000, 3000, 4000])

    print('Initial matrices:\n')
    print('B:')
    B.print_matrix()
    print()
    print('x:')
    x.print_matrix()

    for i in range(10):
        x = dot_product(B, x)
    print('After 10 iterations, x:')
    x.print_matrix()

    for i in range(40):
        x = dot_product(B, x)
    print('After 50 iterations, x:')
    x.print_matrix()

    for i in range(50):
        x = dot_product(B, x)
    print('After 100 iterations, x:')
    x.print_matrix()

main()
