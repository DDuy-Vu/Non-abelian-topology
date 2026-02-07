import cnn
import numpy as np
import sys
import jax.numpy as jnp
import math
import os
from itertools import product


def coor2in(x, y, L):
    return int(((y+L)%L)*(L) + (x+(L))%(L))

def in2coor(i,L):
    x = i % L
    y = (i-x) // L 
    return [x, y]

def compose(p1, p2):
    """Return composition of two permutations: p1 ◦ p2"""
    return [p1[i] for i in p2]

L = None
N_plaquette = None
N = None

plaquette_list = X_list= None
mat_link0 = mat_link1 = mat_link2 = None
gauss_check = plaquette_check1 = plaquette_check2 = None

kernel2 = kernel3 = None
ker_size = None

def gf2_gaussian_elimination(A, b):
    """
    Solve Ax = b over GF(2) using Gaussian elimination.
    A: (k x n) numpy array (k equations, n unknowns)
    b: (k,) numpy array (right-hand side)
    Returns a particular solution x (n,) such that A @ x = b mod 2
    """
    A = A.copy() % 2
    b = b.copy() % 2
    k, n = A.shape

    # Augmented matrix
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(np.uint8)

    row = 0
    for col in range(n):
        pivot = -1
        for r in range(row, k):
            if Ab[r, col] == 1:
                pivot = r
                break

        if pivot == -1:
            continue  # No pivot in this column

        # Swap rows
        Ab[[row, pivot]] = Ab[[pivot, row]]

        # Eliminate below and above
        for r in range(k):
            if r != row and Ab[r, col] == 1:
                Ab[r] ^= Ab[row]

        row += 1

    # Back substitution
    x = np.zeros(n, dtype=np.uint8)
    for i in reversed(range(k)):
        pivot_col = np.where(Ab[i, :-1] == 1)[0]
        if len(pivot_col) == 0:
            if Ab[i, -1] == 1:
                raise ValueError("No solution")
            else:
                continue  # redundant equation
        col = pivot_col[0]
        x[col] = Ab[i, -1]
        for j in range(col + 1, n):
            x[col] ^= Ab[i, j] & x[j]

    return x % 2

def matrix_rank_f2(matrix):
    """Compute the rank of a binary matrix over the field F2."""
    A = np.array(matrix, dtype=np.uint8) % 2  # Ensure binary and np.uint8
    n_rows, n_cols = A.shape
    rank = 0
    row = 0

    for col in range(n_cols):
        # Find a pivot row with a 1 in the current column
        pivot_row = None
        for r in range(row, n_rows):
            if A[r, col] == 1:
                pivot_row = r
                break

        if pivot_row is None:
            continue  # No pivot in this column

        # Swap pivot row into position
        if pivot_row != row:
            A[[row, pivot_row]] = A[[pivot_row, row]]

        # Eliminate all rows below and above
        for r in range(n_rows):
            if r != row and A[r, col] == 1:
                A[r] ^= A[row]  # XOR rows

        row += 1
        rank += 1
        if row == n_rows:
            break

    return rank

def left_inverse_mod2(A_np):
    """
    Given A ∈ F2^{n x m}, compute B ∈ F2^{m x n} such that B @ A ≡ I mod 2
    Assumes A has full column rank (i.e., rank m)
    """
    A = A_np % 2
    n, m = A.shape
    if np.linalg.matrix_rank(A) < m:
        raise ValueError("Matrix A does not have full column rank; left inverse does not exist.")

    B = []
    for i in range(m):
        e = np.zeros(m, dtype=np.uint8)
        e[i] = 1
        x = gf2_gaussian_elimination(A.T, e)  # Solve A.T @ x = e over GF(2)
        B.append(x)

    return np.array(B, dtype=np.uint8)

def generate_mask(N_cell, ker_size, L):

    n_elms = ker_size**2
    ker = np.full((N_cell, N_cell), n_elms)
    
    n = 0
    for del_j, del_i in product(range(-math.floor(ker_size/2), math.ceil(ker_size/2)), range(-math.floor(ker_size/2), math.ceil(ker_size/2))):
        for j, i in product(range(L), range(L)):
            c1 = coor2in(i, j, L)
            i2, j2 = i + del_i, j + del_j
            c2 = coor2in(i2, j2, L)

            ker[c1, c2] = n

        n += 1

    return jnp.array(ker)

def update_globals():
    global N_plaquette, N, plaquette_list, X_list, X_list_r, X_list_b, X_list_g, left_triangles, right_triangles
    global kernel2, kernel3, transform_matrix, point_group, inverse_matrix
    global adjacent_matrix, path_matrix, translation_cell, translation_site, kx, ky

    if L is not None:
        N_plaquette = L**2
        N = 3*L**2

        plaquette_list = jnp.array([[3*coor2in(i,j,L), 3*coor2in(i+1,j,L)+2, 3*coor2in(i+1,j,L)+1, 
                    3*coor2in(i+1,j+1,L), 3*coor2in(i+1,j+1,L)+2, 3*coor2in(i,j,L)+1] for j in range(L) for i in range(L) ])
        
        left_triangles = jnp.array([[3*coor2in(i,j,L), 3*coor2in(i+1,j,L)+1, 3*coor2in(i+1,j+1,L)+2] for j in range(L) for i in range(L) ])
        
        right_triangles = jnp.array([[3*coor2in(i+1,j,L)+2, 3*coor2in(i+1,j+1,L), 3*coor2in(i,j,L)+1] for j in range(L) for i in range(L) ])

        X_list = jnp.array([[3*coor2in(i,j,L)+2, 3*coor2in(i,j-1,L)+1, 3*coor2in(i+1,j,L)+0, 
                    3*coor2in(i+2,j+1,L) + 2, 3*coor2in(i+1,j+1,L)+1, 3*coor2in(i,j+1,L)+0] for j in range(L) for i in range(L) ])
    
        translation_site = []
        translation_cell = []
        for del_j, del_i in product(range(L), range(L)):
            temp_site = []
            temp = []
            for j, i in product(range(L), range(L)):
                i2 = i + del_i
                j2 = j + del_j
                c = coor2in(i2, j2, L)
                temp_site.extend([3*c, 3*c + 1, 3*c + 2])
                temp.append(c)
            
            translation_site.append(temp_site)
            translation_cell.append(temp)
        
        translation_site, translation_cell = jnp.array(translation_site), jnp.array(translation_cell)

        kernel2 = generate_mask(L**2, 2, L)
        kernel3 = generate_mask(L**2, 3, L)


        C6_matrix = np.zeros(N, dtype = int)
        for j, i in product(range(L), range(L)):
            i2 = i - j
            j2 = i
            plaquette_index = int(((j+L)%L)*L + (i+L)%L)
            rotated_plaquette = int(((j2+L)%L)*L + (i2+L)%L)
            C6_matrix[3 * rotated_plaquette + 0] = 3*plaquette_index + 1
            
            rotated_plaquette = int(((j2+L)%L)*L + (i2+L+1)%L)
            C6_matrix[3 * rotated_plaquette + 2] = 3*plaquette_index + 0

            rotated_plaquette = int(((j2+L-1)%L)*L + (i2+L)%L)
            C6_matrix[3 * rotated_plaquette + 1] = 3*plaquette_index + 2


        point_group = [np.arange(N)]
        C = np.arange(N)
        for _ in range(5):
            C = compose(C6_matrix, C)
            point_group.append(C)

        point_group = jnp.array(point_group)

        transform_matrix = np.full((N, len(X_list)), 0, dtype=jnp.int8)
        for i in range(len(X_list)):
            transform_matrix[X_list[i],i] = 1

        if L % 3 == 0:
            X_list_r = jnp.array([coor2in(i,j,L) for j in range(L) for i in range(L) if (i+j)%3 == 0])
            X_list_g = jnp.array([coor2in(i,j,L) for j in range(L) for i in range(L) if (i+j)%3 == 1])
            X_list_b = jnp.array([coor2in(i,j,L) for j in range(L) for i in range(L) if (i+j)%3 == 2])

            a = np.sort([X_list_r[-1].item(), X_list_g[-1].item(), X_list_b[-1].item()])

            inverse_matrix = left_inverse_mod2( np.delete(transform_matrix, a, axis = -1) )
            inverse_matrix = np.insert(inverse_matrix, [a[0], a[1]-1, a[2]-2], np.zeros((3, N)), axis=0)

        else:

            inverse_matrix = left_inverse_mod2(np.delete(transform_matrix,[N_plaquette-1], axis = -1))
            inverse_matrix = np.insert(inverse_matrix, [N_plaquette-1], np.zeros((1, N)), axis=0)

        coordinates = np.argwhere(inverse_matrix == 1)

        path_matrix = np.zeros((N_plaquette, 6, 6))
        adjacent_matrix = np.full((N_plaquette, 6) , -1)


        displacement = [[-1, -1], [0, -1], [1, 0], [1, 1], [0, 1], [-1, 0]]

        for i in range(N_plaquette):
            ## Adjacent X_rotate
            x, y = in2coor(i , L)
            u = plaquette_list[i]

            for (k, d) in enumerate(displacement):

                i_neighbor = coor2in(x+d[0], y+d[1], L)
                if i_neighbor < i and i_neighbor not in adjacent_matrix[i, :]:
                    path_matrix[i, k, :] = transform_matrix[u, i_neighbor]
                adjacent_matrix[i, k] = i_neighbor

        path_matrix = jnp.array(path_matrix)
        adjacent_matrix = jnp.array(adjacent_matrix)


        basis = np.array([[1.0, 0.0], [-0.5, np.sqrt(3.) / 2.0],])
        cell = np.array([basis[0] / 2.0, (basis[0]+basis[1])/2.0, [0., 0.]])
        kx, ky = [], []
        for j, i in product(range(L), range(L)):
            kx.extend([np.exp(2j*np.pi*(i+cell[0, 0])/L),  np.exp(2j*np.pi*(i+cell[1, 0])/L), np.exp(2j*np.pi*(i+cell[2, 0])/L)])
            ky.extend([np.exp(2j*np.pi*(j+cell[0, 1])/L),  np.exp(2j*np.pi*(j+cell[1, 1])/L), np.exp(2j*np.pi*(j+cell[2, 1])/L)])

        kx, ky = jnp.array(kx), jnp.array(ky)