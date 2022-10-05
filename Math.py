import numpy as np
import math 

def zeros_matrix(rows, cols):
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def transposeMatrix(m):
    return map(list,zip(*m))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = list(transposeMatrix(cofactors))
    # print(list(cofactors))
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors



def multiply_matrix(matrix1, matrix2):
    matrix = [
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
    ]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                matrix[i][j] += np.float64(matrix1[i][k] * matrix2[k][j])
            
    return matrix


def multiply_matrix_vector(matrix, vector):
    result = [0,0,0,0]
    for i in range(4):
        for j in range(4):
            result[i] += matrix[i][j] * vector[j]
    return result

def matrix_substract(matrix1, matrix2):
    rows1 =len(matrix1)
    rows2 =len(matrix2)
    cols1 =len(matrix1[0])
    cols2 =len(matrix2[0])

    if rows1 != rows2 or cols1 != cols2:
        print("Error: matrices must be of same size")
        return
    result = zeros_matrix(rows1,cols1)
    for row in range(0,rows1):
        for col in range(0,cols1):
            result[row][col]= matrix1[row][col] - matrix2[row][col]
    return result

def vector_substract(vector1, vector2):
    result = []
    for i in range(len(vector1)):
        result.append(vector1[i] - vector2[i])
    return result

def cross_product(v1, v2):
    return [v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0]]

def negative_vector(v):
    r = []
    for x in v:
        r.append(-x)
    return r

def dot_product(a, b):
    return sum([i*j for (i, j) in zip(a, b)])

# get matrix norm 
def matrix_norm(m):
    return math.sqrt(sum([i*i for i in m]))

def matrix_add(m1, m2):
    return [i+j for (i, j) in zip(m1, m2)]

def vector_add(v1, v2):
    return [i+j for (i, j) in zip(v1, v2)]

def number_multiply_matrix(n, m):
    return [n*i for i in m]

def normalize(v):
    norm = matrix_norm(v)
    return [i/norm for i in v]

def vector_Multiply(v1, v2):
    return [i*j for (i, j) in zip(v1, v2)]