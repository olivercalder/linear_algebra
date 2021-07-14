class Matrix:

    def __init__(self, matrix=None, width=None):
        if matrix == None:
            self.empty()
        elif width != None:
            height = matrix
            self.empty(height, width)
        else:
            self.height = len(matrix)
            self.width = len(matrix[0])
            self.matrix = matrix

    def empty(self, height=0, width=0):
        self.height = height
        self.width = width
        self.matrix = []
        for h in range(self.height):
            self.matrix.append([])
            for w in range(self.width):
                self.matrix[h].append(None)
        return self

    def zero(self, height=None, width=None):
        if height != None:
            self.height = height
            self.width = width
        if width == None:
            self.width = self.height
        self.matrix = []
        for h in range(self.height):
            self.matrix.append([])
            for w in range(self.width):
                self.matrix[h].append(0)
        return self

    def identity(self, order, order2=None):
        if order2 != None and order2 != order:
            return None
        else:
            self.zero(order, order)
            for i in range(1, order + 1):
                self.set(i, i, 1)
        return self

    def __repr__(self):
        return 'Matrix({})'.format(self.matrix)

    def __str__(self):
        return self.get_matrix_string()

    def __lt__(self, other):
        if self.det() < other.det():
            return True
        else:
            return False

    def __le__(self, other):
        if self == other:
            return True
        elif self < other:
            return True
        else:
            return False

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif self.width != other.width or self.height != other.height:
            return False
        else:
            equal = True
            for row in range(self.height):
                for column in range(self.width):
                    if self.matrix[row][column] != other.matrix[row][column]:
                        equal = False
            return equal

    def __gt__(self, other):
        if self.det() > other.det():
            return True
        else:
            return False

    def __ge__(self, other):
        if self == other:
            return True
        elif self > other:
            return True
        else:
            return False

#   def __getattr__(self, name):        Implement the following for things like:
#       pass                            A.R1 = -1*A.R1 + 3*A.R2

#   def __getattribute(self, name):
#       pass

#   def __setattr__(self, name, value):
#       pass

#   def __get__(self, instance, owner):
#       pass

#   def __set__(self, instance, owner):
#       pass

    def __len__(self):
        return self.height * self.width

    def __getitem__(self, item):
        if type(item) == type(tuple([])):
            return self.get(item[0], item[1])
        elif type(item) == type('Rx'):
            if item[0].upper() == 'R':
                return self.get_row(int(item[1:]))
            elif item[0].upper() == 'C':
                return self.get_column(int(item[1:]))
        elif type(item) == type(0):
            row = ((item - 1) // self.width) + 1
            column = ((item - 1) % self.width) + 1
            return self.get(row, column)

    def __setitem__(self, item, value):
        if type(item) == type(tuple([])):
            self.set(item[0], item[1], value)
        elif type(item) == type('Rx'):
            if item[0].upper() == 'R':
                self.set_row(int(item[1:]), value)
            elif item[0].upper() == 'C':
                return self.set_column(int(item[1:]))
        elif type(item) == type(0):
            row = (item - 1) // self.width
            column = ((item - 1) % self.width) + 1
            self.set(row, column, self)
        else:
            return None
        return value

#   def __iter__(self):
#       pass

    def __add__(self, other):
        if self.order() != other.order():
            return None
        else:
            new_matrix = self.copy()
            for row in range(1, self.height + 1):
                for column in range(1, self.width + 1):
                    new_value = self.get(row, column) + other.get(row, column)
                    if abs(new_value) < 10**-13:
                        new_value = 0
                    if abs(new_value) == 0:
                        new_value = 0
                    new_matrix.set(row, column, new_value)
        return new_matrix

    def __radd__(self, other):
        return NotImplemented

    def __iadd__(self, other):
        self = self + other
        return self
        
    def __sub__(self, other):
        if self.order() != other.order():
            return None
        else:
            new_matrix = self.copy()
            for row in range(1, self.height + 1):
                for column in range(1, self.width + 1):
                    new_value = self.get(row, column) - other.get(row, column)
                    if abs(new_value) < 10**-13:
                        new_value = 0
                    if abs(new_value) == 0:
                        new_value = 0
                    new_matrix.set(row, column, new_value)
        return new_matrix

    def __rsub__(self, other):
        return NotImplemented

    def __isub__(self, other):
        self = self - other
        return self

    def __mul__(self, other):
        if type(other) == type(0) or type(other) == type(0.0):
            new_matrix = self.copy()
            for row in range(1, self.height + 1):
                for column in range(1, self.width + 1):
                    new_matrix.set(row, column, self.get(row, column) * other)
                    if abs(new_matrix.get(row, column)) == 0:
                        new_matrix.set(row, column, 0)
            return new_matrix
        elif self.order()[1] != other.order()[0]:
            return None
        elif type(other) == type(self):
            new_matrix = Matrix(self.height, other.width)
            for i in range(1, self.height + 1):
                for j in range(1, other.width + 1):
                    total = 0
                    for k in range(1, other.height + 1):
                        total += self.get(i, k) * other.get(k, j)
                    new_matrix.set(i, j, total)
            return new_matrix

    def __rmul__(self, other):
        # multiplying matrix by a scalar int or float
        new_matrix = self.copy()
        for row in range(1, self.height + 1):
            for column in range(1, self.width + 1):
                new_value = self.get(row, column) * other
                if abs(new_value) == 0:
                    new_value = 0
                new_matrix.set(row, column, new_value)
        return new_matrix

    def __imul__(self, other):
        self = self * other
        return self

    def __matmul__(self, other):
        return NotImplemented
        # cross product

    def __rmatmul__(self, other):
        return NotImplemented

    def __imatmul__(self, other):
        return NotImplemented
        # cross product

    def __truediv__(self, other):
        new_matrix = self.copy()
        for row in range(1, self.height + 1):
            for column in range(1, self.width + 1):
                new_value = self.get(row, column) / other
                if abs(new_value) == 0:
                    new_value = 0
                new_matrix.set(row, column, new_value)
        return new_matrix

    def __rtruediv__(self, other):
        return NotImplemented

    def __itruediv__(self, other):
        self = self / other
        return self

    def __floordiv__(self, other):
        new_matrix = self.copy()
        for row in range(1, self.height + 1):
            for column in range(1, self.width + 1):
                new_value = self.get(row, column) // other
                if abs(new_value) == 0:
                    new_value = 0
                new_matrix.set(row, column, new_value)
        return new_matrix

    def __rfloordiv__(self, other):
        return NotImplemented

    def __ifloordiv__(self, other):
        self = self / other
        return self

    def __mod__(self, other):
        new_matrix = self.copy()
        for row in range(1, self.height + 1):
            for column in range(1, self.width + 1):
                new_value = self.get(row, column) % other
                if abs(new_value) == 0:
                    new_value = 0
                new_matrix.set(row, column, new_value)
        return new_matrix

    def __rmod__(self, other):
        return NotImplemented

    def __imod__(self, other):
        for row in range(1, self.height + 1):
            for column in range(1, self.width + 1):
                self = self % other
        return self

    def __divmod__(self, other):
        return NotImplemented

    def __rdivmod__(self, other):
        return NotImplemented

    def __idivmod__(self, other):
        return NotImplemented

    def __pow__(self, other):
        if other < 0:
            new_matrix = self.inverse()
        else:
            new_matrix = self.copy()
        multiplicand = new_matrix.copy()
        for i in range(1, int(abs(other))):
            new_matrix *= multiplicand
        return new_matrix

    def __rpow__(self, other):
        return NotImplemented

    def __ipow__(self, other):
        self = self**other
        return self

    def __lshift__(self, other):
        return NotImplemented

    def __rlshift__(self, other):
        return NotImplemented

    def __ilshift__(self, other):
        return NotImplemented

    def __rshift__(self, other):
        return NotImplemented

    def __rrshift__(self, other):
        return NotImplemented

    def __irshift__(self, other):
        return NotImplemented

    def __and__(self, other):
        return NotImplemented

    def __rand__(self, other):
        return NotImplemented

    def __iand__(self, other):
        return NotImplemented

    def __xor__(self, other):
        return NotImplemented

    def __rxor__(self, other):
        return NotImplemented

    def __ixor__(self, other):
        return NotImplemented

    def __or__(self, other):
        return NotImplemented

    def __ror__(self, other):
        return NotImplemented

    def __ior__(self, other):
        return NotImplemented

    def __neg__(self):
        return -1 * self

    def __abs__(self):
        if self.height == 1 or self.width == 1:
            sum_of_squares = 0
            for row in range(1, self.height + 1):
                for column in range(1, self.width + 1):
                    sum_of_squares += (self.get(row, column))**2
            return sum_of_squares**(1/2)
        else:
            return self.det()

    def __invert__(self):
        return self.inverse()

    def order(self):
        return tuple([self.height, self.width])

    def is_square(self):
        return self.order()[0] == self.order()[1]

    def set(self, row, column, number):
        row = row - 1
        column = column - 1
        self.matrix[row][column] = number

    def get(self, row, column):
        row = row - 1
        column = column - 1
        return self.matrix[row][column]

    def get_row_list(self, row):
        row_data = []
        for i in range(1, self.width + 1):
            row_data.append(self.get(row, i))
        return row_data

    def get_column_list(self, column):
        column_data = []
        for i in range(1, self.height + 1):
            column_data.append(self.get(i, column))
        return column_data

    def get_row(self, row):
        return Matrix([self.get_row_list(row)])

    def get_column(self, column):
        return Matrix([self.get_column_list(column)]).flip()

    def get_submatrix(self, upperleft, lowerright):
        new_matrix = []
        for j in range(upperleft[0], lowerright[0] + 1):
            new_row = []
            for k in range(upperleft[1], lowerright[1] + 1):
                new_row.append(self.get(j, k))
            new_matrix.append(new_row)
        return Matrix(new_matrix)

    def set_row(self, row, data):
        if type(data) == type(list()):
            if self.width != 0 and self.width != len(data):
                print('Error: Cannot set row. Length does not match.')
                return None
            else:
                for i in range(1, self.width + 1):
                    self.set(row, i, data[i - 1])
        elif type(data) == type(Matrix()):
            if self.width != 0 and self.width != data.width:
                if self.width == data.height and data.width == 1:
                    data = data.flip()
                else:
                    print('Error: Cannot set row. Size does not match.')
                    return None
            if data.height == 1:
                for i in range(1, self.width + 1):
                    self.set(row, i, data.get(1, i))
        else:
            print('Error: Cannot set row. Type does not match.')
            return None

    def set_column(self, column, data):
        if type(data) == type(list()):
            if self.height != 0 and self.height != len(data):
                print('Error: Cannot set column. Length does not match.')
                return None
            else:
                for i in range(1, self.height + 1):
                    self.set(i, column, data[i - 1])
        elif type(data) == type(Matrix()):
            if self.height != 0 and self.height != data.height:
                if self.height == data.width and data.height == 1:
                    data = data.flip()
                else:
                    print('Error: Cannot set column. Size does not match.')
                    return None
            if data.width == 1:
                for i in range(1, self.height + 1):
                    self.set(i, column, data.get(i, 1))
        else:
            print('Error: Cannot set column. Type does not match.')
            return None

    def add_row(self, data):
        if (self.order() == (0, 0)) or (type(data) == type(list()) and self.width == len(data)) or (type(data) == type(Matrix()) and self.width == data.width):
            self.height += 1
            self.matrix.append([])
            if self.width == 0:
                self.width = len(data)
            for i in range(1, self.width + 1):
                self.matrix[self.height - 1].append(None)
            self.set_row(self.height, data)
            return self
        else:
            print('Error: Cannot add row. Length or type does not match.')
            return None

    def add_column(self, data):
        if (self.order() == (0, 0)) or (type(data) == type(list()) and self.height == len(data)) or (type(data) == type(Matrix()) and self.height == data.height):
            self.width += 1
            if self.height == 0:
                self.height = len(data)
                for i in range(1, self.height + 1):
                    self.matrix.append([])
            for i in range(1, self.height + 1):
                self.matrix[i - 1].append(None)
            self.set_column(self.width, data)
            return self
        else:
            print('Error: Cannot add column. Length or type does not match.')
            return None

    def minor(self, i, j):
        ij_minor = Matrix()
        for r in range(1, self.height + 1):
            if r != i:
                new_row = []
                for c in range(1, self.width + 1):
                    if c != j:
                        new_row.append(self[r,c])
                ij_minor.add_row(new_row)
        return ij_minor

    def det(self):
        A = self
        if A.height != A.width:
            return None
        elif A.height == 1 and A.width == 1:
            return A[1,1]
        else:
            determinant = 0
            for j in range(1, A.width + 1):
                if A[1,j] != 0:
                    determinant += (-1)**(j+1) * A[1,j] * A.minor(1,j).det()
            return determinant

    def inverse(self):
        if self.order()[0] != self.order()[1]:
            print('Error: Cannot invert. Must be nxn matrix.')
            return None
        elif self.det() == 0:
            print('Error: Cannot invert. Determinant = 0.')
            return None
        else:
            A = self.copy()
            degree = A.order()[0]
            Aaug = A.conjoin(Matrix().identity(degree))
            Aaug = Aaug.rref()
            Ainv = Aaug.get_submatrix((1, 1+degree), (degree, 2*degree))
            zero = Matrix().zero(1, degree)
            for row in range(degree):
                if Aaug.get_submatrix((1, 1), (degree, degree)) == zero and Ainv.get_submatrix((1, 1), (degree, degree)) != zero:
                    print('Error: Cannot invert. No solution to rref(A|I).')
                    return None
            return Ainv

    def copy(self):
        A = Matrix()
        for i in range(1, self.height + 1):
            A.add_row(self.get_row(i))
        return A
    
    def flip(self):
        A = Matrix().empty(self.width, self.height)
        for i in range(1, A.height + 1):
            for j in range(1, A.width + 1):
                A.set(i, j, self.get(j, i))
        return A

    def conjoin(self, other):
        A = self.copy()
        for i in range(1, other.width + 1):
            A.add_column(other.get_column(i))
        return A

    def R(self, row): # deprecated in favor of A['R1'], but still better when referring to rows using variables
        row_list = self.get_row(row)
        return self.get_row(row)

    def set_R(self, row, matrix): # deprecated in favor of A['R1'], but still better when referring to rows using variables
        self.set_row(row, matrix)

    def swap_R(self, row1, row2):
        tmp = self.get_row(row1)
        self.set_row(row1, self.get_row(row2))
        self.set_row(row2, tmp)

    def rref(self):
        A = self.copy()
        n = 1
        m = 1
        while n <= A.height and m <= A.width:
            i = n
            while i <= A.height and A.get(i, m) == 0:
                i += 1
            if i > A.height:
                m += 1 # Shifts the start index over one column, but does not shift down a row
            else:
                A.swap_R(n, i)
                A['R{}'.format(n)] /= A[n,m] # Old method: A.set_R(n, 1/A.get(n, m) * A.R(n))
                for j in range(1, A.height + 1):
                    if j != n and A.get(j, m) != 0:
                        A['R{}'.format(j)] -= A[j,m] * A['R{}'.format(n)] # Old method: A.set_R(j, A.R(j) - A.get(j, m) * A.R(n))
                m += 1 # Shifts the start index over one column
                n += 1 # Shifts the start index down one row
        return A

    def get_row_string(self, row):
        row_string = '( '
        for column in range(1, self.width + 1):
            row_string = row_string + '{:^5.4g}'.format(self.get(row, column)) + ' '
        row_string = row_string + ')'
        return row_string

    def get_column_string(self, column):
        column_string = ''
        for row in range(1, self.height + 1):
            row_string = '( ' + '{:^5.4g}'.format(self.get(row, column)) + ' )'
            column_string = column_string + row_string + '\n'
        column_string.rstrip('\n')
        return column_string

    def get_matrix_string(self):
        matrix_string = ''
        for row in range(1, self.height + 1):
            matrix_string += self.get_row_string(row) + '\n'
        matrix_string.rstrip('\n')
        return matrix_string

    def print_row(self, row):
        row_string = self.get_row_string(row)
        print(row_string)

    def print_column(self, column):
        column_string = self.get_column_string(column)
        print(column_string)

    def print_matrix(self):
        matrix_string = self.get_matrix_string()
        print(matrix_string)

def test():
    test = Matrix([
                   [ 1,  2,  2, -5,  6],
                   [-1, -2, -1,  1, -1],
                   [ 4,  8,  5, -8,  9],
                   [ 3,  6,  1,  5, -7]
                   ])
    print(test)
    print('Slot (3, 3):', test[3, 3])
    test = test.rref()
    print('rref:')
    print(test)
    print()
    print('New Matrix A:')
    A = Matrix([
                [22, 13, 8, 3],
                [-16, -3, -2, -2],
                [8, 9, 7, 2],
                [5, 4, 3, 1]
              ])
    print(A)
    print('rref(A):')
    print(A.rref())
    print('A^-1:')
    print(A**-1)
    print('A[3,3]:')
    print(A[3,3])
    print("\nA['R2']:")
    print(A['R2'])
    print('A[5]:')
    print(A[5])
    print("\nA['R2'] = A['R2'] + 2 * A['R3']")
    print('A after operation:')
    A['R2'] = A['R2'] + 2 * A['R3']
    print(A)

if __name__ == '__main__':
    test()
