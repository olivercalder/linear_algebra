class Matrix:

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.matrix = []
        for h in range(self.height):
            self.matrix.append([])
            for w in range(self.width):
                self.matrix[h].append(None)

    def set(self, row, column, number):
        row = row - 1
        column = column - 1
        self.matrix[row][column] = number

    def get(self, row, column):
        row = row - 1
        column = column - 1
        return self.matrix[row][column]

    def get_row(self, row):
        row_data = []
        for i in range(1, self.width + 1):
            row_data.append(self.get(row, i))
        return row_data

    def get_column(self, column):
        column_data = []
        for i in range(1, self.height + 1):
            column_data.append(self.get(i, column))
        return column_data

    def set_row(self, row, data_list):
        if self.width != len(data_list):
            print('Row width does not match matrix width! Exiting...')
            return None
        else:
            for i in range(1, self.width + 1):
                self.set(row, i, data_list[i - 1])

    def set_column(self, column, data_list):
        if self.height != len(data_list):
            print('Column height does not match matrix height! Exiting...')
            return None
        else:
            for i in range(1, self.height + 1):
                self.set(i, column, data_list[i - 1])

    def add_row(self, data_list):
        self.height += 1
        self.matrix.append([])
        for i in range(self.width):
            self.matrix[self.height - 1].append(None)
        self.set_row(self.height, data_list)

    def add_column(self, data_list):
        self.width += 1
        for i in range(1, self.height + 1):
            self.matrix[i].append(None)
        self.set_column(self.width, data_list)

    def build(self, data):
        for i in range(len(data)):
            self.set_row(i + 1, data[i])

    def copy(self):
        new_matrix = Matrix(self.height, self.width)
        new_matrix.build(self.matrix.copy())
        return new_matrix
    
    def flip(self):
        new_matrix = Matrix(self.width, self.height)
        for i in range(1, new_matrix.height + 1):
            new_matrix.set_row(i, self.get_column(i))
        return new_matrix

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
