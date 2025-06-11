import numpy as np
import math

class Verifier():
    '''
    Inputs:
    array --> The sudoku array
    order --> The oder of the sudoku array, for example, if the sudoku array is 9 by 9, then order = 9

    Methods:
    verfify_rows() --> verfifies whether all the rows have unique number from 1 to 9; returns True or False
    verfify_colums() --> verfifies whether all the columns have unique number from 1 to 9; returns True or False
    verfify_cells() --> verfifies whether all the 3 by 3 cells have unique number from 1 to 9; returns True or False
    '''   
    def __init__(self, array, order):
        if array.shape != (order, order):
            raise ValueError(f"Input array must have shape ({order}, {order}), but got {array.shape}")
        if math.isqrt(order) ** 2 != order: # isqrt() returns the floor of a square root, which is always an integer
            raise ValueError(f"Order must be a perfect square, but got {order}")
        self.array = array
        self.order = order

    def verify_rows(self):
        for i in range(self.array.shape[0]): # loop for all rows
            row = self.array[i] # extract the i row, i range from 0 to 8
            if len(np.unique(row)) != row.size:
                return False # np.unique returns the unique elements in an array; return False if not all the elements are unique in a row
        return True # return True if the loop is not interrupted
    
    def verify_columns(self):
        for i in range(self.array.shape[1]): # loop for all colums
            column = self.array[:,i]
            if len(np.unique(column)) != column.size:
                return False
        return True
    
    def verify_cells(self):
        cell_len = math.isqrt(self.order) # cell lenght per axis
        num_cells = int(self.order / cell_len) # number of cells per axis
        for i in range(num_cells):
            for j in range(num_cells):
                row_start = i * cell_len
                column_start = j * cell_len
                cell = self.array[row_start : row_start + cell_len, column_start : column_start + cell_len]
                #print(cell)
                if len( np.unique( cell.flatten() ) ) != cell.size:
                    return False             
        return True
