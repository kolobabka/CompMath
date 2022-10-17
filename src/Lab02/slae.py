#####           Imports

import numpy as np
from abc import ABC, ABCMeta, abstractmethod, abstractproperty

####            Abstract class 'Callable'
class ICallable (ABC):

    @abstractmethod
    def __call__(self):
        pass

#####           Class norm

class INorm (ICallable, ABC):
    norm = 0

    def __init__(self, norm):
        self.norm = norm
        
class VecNorm (INorm):

    def __call__(self, vec):
        return self.norm(vec)

class MatrixNorm (INorm):

    def __call__(self, matrix):
        return self.norm(matrix)

#####           Supporting functions for solving the task

def find_max_eig (matrix):
    dim = matrix.shape[0]
    x = np.random.rand(dim)
    num_iters = 100
    for i in range(num_iters):
        x = np.dot(matrix, x) / VecNorms[0](x)
    return np.dot(np.dot(x.T, matrix), x) / np.dot(x.T, x)

def find_min_eig (matrix):
    return 1 / find_max_eig(np.linalg.inv(matrix))

def is_LU_possible(matrix):
    dim = matrix.shape[0]
    for i in range(dim):
        if np.linalg.det(matrix[:i, :i]) == 0:
            return False
    return True    

def get_LU (matrix):
    dim = matrix.shape[0]
    L_part = np.eye(dim)
    U_part = np.zeros((dim, dim))
    
    for i in range(dim):
        for j in range(dim):
            if i > j:
                L_part[i][j] = (matrix[i][j] - sum(np.dot(L_part[i:i+1, 0:j], U_part[0:j, j:j+1]))) / U_part[j][j]
            else:
                U_part[i][j] = matrix[i][j] - sum(np.dot(L_part[i:i+1, 0:i], U_part[0:i, j:j+1]))
    return L_part, U_part

def get_l_solution(L_part, rhs):
    sol = np.zeros((L_part.shape[0], 1))
    sol[0] = rhs[0] / L_part[0][0]
    for i in range(1, len(sol)):
        sol[i] = (rhs[i] - sum(np.dot(L_part[i:i+1, :i], sol[:i]))) / L_part[i][i]
    return sol

def get_u_solution(U_part, sol):
    U_part_dim = U_part.shape[0]

    x = np.zeros((U_part_dim, 1))
    x[-1] = sol[-1] / U_part[-1][-1]
    for i in range(U_part_dim - 1, -1, -1):
        x[i] = (sol[i] - sum(np.dot(U_part[i:i+1, i+1:], x[i+1:]))) / U_part[i][i]
    return x

def over_relaxation_is_available(matrix):
    for i in range(matrix.shape[0]):
        if np.linalg.det(matrix[:i, :i]) <= 0:
            return False
    return True

def over_relaxation(matrix, rhs, epsilon=1e-6):
    dim = matrix.shape[0]
    
    D_part = np.zeros([dim, dim])
    L_part = np.zeros([dim, dim])
    U_part = np.zeros([dim, dim])
    
    for i in range(0, dim):
        for j in range(0, dim):
            if i < j:
                U_part[i, j] = matrix[i, j]
            elif i > j:
                L_part[i, j] = matrix[i, j]
            else:
                D_part[i, j] = matrix[i, j]
    
    w = np.random.rand() + 1.01 # the random number in range (1, 2)
    tmp = np.linalg.inv(w * L_part + D_part)
    u_k = np.matmul(-tmp, D_part * (w - 1) + U_part * w)
    
    num_iters = 0
    x = (np.random.rand(1, dim)).T
    norm = VecNorms[2]
    while norm(rhs - np.matmul(matrix, x)) >= epsilon:
        x = np.matmul(w * tmp, rhs) + np.matmul(u_k, x) 
        num_iters += 1

    return (x, num_iters)

#### Initialization

dim = 10
rhs = np.array([[1 / i] for i in range(1, dim + 1)])
matrix = np.zeros((dim, dim)) 

for i in range(dim): # Fill the matrix from point 'k'
    for j in range(dim):
        if i == j:
            matrix[i, j] = 1
        else:
            matrix[i, j] = 1 / (i + j + 2)

VecNorms = [] #Norm for vectors
VecNorms.append(VecNorm(lambda vec: np.abs(vec).max()))
VecNorms.append(VecNorm(lambda vec: np.abs(vec).sum()))
VecNorms.append(VecNorm(lambda vec: np.sqrt(np.dot(vec.T, vec)).item()))

MatrixNorms = [] #Norm for matrixes
MatrixNorms.append(MatrixNorm(lambda matrix: np.abs(np.abs(matrix).sum(axis=1, dtype='float')).max()))
MatrixNorms.append(MatrixNorm(lambda matrix: np.abs(np.abs(matrix).sum(axis=0, dtype='float')).max()))
MatrixNorms.append(MatrixNorm(lambda matrix: np.sqrt(find_max_eig(np.matmul(matrix.T, matrix)))))


def main ():

    eigenvec = np.linalg.eigvals(matrix)

    real_min_eigval = min(eigenvec)
    real_max_eigval = max(eigenvec)

    iter_max_eigval = find_max_eig(matrix); 
    iter_min_eigval = find_min_eig(matrix)

    print ('Iteratively calculated max eigenvalue: ' + str(real_max_eigval))
    print ('Iteratively calculated min eigenvalue: ' + str(real_min_eigval))

    print('Error of iterative method of computing max eigenvalue: ' + \
           str(np.abs((iter_max_eigval - real_max_eigval))))

    print('Error of iterative method of computing min eigenvalue: ' + \
           str(np.abs((iter_min_eigval - real_min_eigval))))

    print ("--------------------------")

    for i in range (3):
        print ('Condition number with matrix norms #' + str(i) + ': ' + \
               str(MatrixNorms[i](matrix) * MatrixNorms[i](np.linalg.inv(matrix))))

    print ("--------------------------")

    print ('Is LU decomposition is possible?')
    if (is_LU_possible(matrix)):
        print ('\t Yes, it is')
    else:
        print ('\t No, it isn\'t')

    L, U = get_LU(matrix)
    y = get_l_solution(L, rhs)
    ans = get_u_solution(U, y)
    print("The solution via LU-decompisition:")
    print(ans)

    error_vec = np.abs(np.matmul(matrix, ans) - rhs)
    print("Errors with different norms:")
    for i in range (3):
        print ('\tThe ' + str(i + 1) + 'th norm of error\'s vector: ' + str(VecNorms[i](error_vec)))

    print ("--------------------------")

    print ('Is over relaxation is possible?')
    if (over_relaxation_is_available(matrix)):
        print ('\t Yes, it is')
    else:
        print ('\t No, it isn\'t')
    
    solution, num_iters = over_relaxation(matrix, rhs)
    print("The solution via over relaxation:")
    print(solution)
    print("Number of iterations: " + str(num_iters))

    error_vec = np.abs(np.matmul(matrix, solution) - rhs)
    print("Errors with different norms:")
    for i in range (3):
        print ('\tThe ' + str(i + 1) + 'th norm of error\'s vector: ' + str(VecNorms[i](error_vec)))

main()