# %%
import numpy as np
import scipy.linalg
from thewalrus.decompositions import williamson, blochmessiah
import strawberryfields as sf
from tenpy.networks import site
from tenpy.linalg import np_conserved as npc

# %%
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    """
    Generates a tridiagonal matrix from 3 1-D arrays. such that b is the main diagonal, a is the lower diagonal and c the upper
    diagonal.
    one can also change where the diagonal should be. NB: Function does not check if the input has the right form.
    Need to make sure when passing
    
    Input Parameters: 
    a: 1-D array or list containing the elements for diagonal k1
    b: 1-D array or list containing the elements for diagonal k2
    c: 1-D array or list containing the elements for diagonal k3
    k1: Specification which diagonal to write a on. Default -1
    k2: Specification which diagonal to write b on. Default 0
    k3: Specification which diagonal to write c on. Default 1
    """
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def Random_NNhamiltonian(N):
    """
    Generates a NxN random Nearest neighbour hamiltonian matrix in the ladder operator basis. 
    
    Input Parameters: 
    N: number of particles/ modes
    
    Return:
    H = The hamiltonian matrix 
    """
    
    # Generate random matrices for the Hamiltonian
    ud = np.random.rand(N-1) + 1j * np.random.rand(N-1)
    d = np.random.rand(N) + 1j * np.random.rand(N)
    bd = np.random.rand(N-1) + 1j * np.random.rand(N-1)
    
    # Create the matrices A and B using the Tridiagonal function
    A = tridiag(bd, d, ud)
    A = (A + A.conj().T) / 2.  # Symmetrize A
    B = tridiag(bd, np.zeros(N, dtype=np.complex128), -bd)
    
    # Initialize the Hamiltonian matrix
    H = np.zeros((2*N, 2*N), dtype=np.complex128)
    
    # Fill the blocks of the Hamiltonian matrix obeying the Anticommutation relations
    H[:N, :N] = -A.conj()
    H[:N, N:] = -B.conj()
    H[N:, :N] = B
    H[N:, N:] = A
    
    return H

def random_fermionic_hamiltonian(N):
    """
    Generates a NxN random fermionic hamiltonian matrix in the ladder operator basis. 
    
    Input Parameters: 
    N: number of particles/ modes
    
    Return:
    H = The hamiltonian matrix 
    """

    A = np.random.rand(N, N) + 1j * np.random.rand(N,N)
    A = (A + A.conj().T) / 2.  # Symmetrize A
    B = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    # Make it skew-symmetric using the conjugate transpose
    B = (B - B.T) / 2
    
    # Initialize the Hamiltonian matrix
    H = np.zeros((2*N, 2*N), dtype=np.complex128)
    
    # Fill the blocks of the Hamiltonian matrix obeying the Anticommutation relations
    H[:N, :N] = A
    H[:N, N:] = -B.conj()
    H[N:, :N] = B
    H[N:, N:] = -A.conj()
    
    return H

def create_skew_symmetric(N):
    """
    Generates a NxN random skew_symmetric matrix 

    Input Parameters: 
    N: dimension of the matrix

    Return:
    B: A random skew symmetric of dimension NxN
    """

    B = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    # Make it skew-symmetric using the conjugate transpose
    B = (B - B.T) / 2

    return B

def random_symmetric_matrix(N):
    
    """
    Generates a NxN random symmetric matrix 

    Input Parameters: 
    N: dimension of the matrix

    Return:
    A: A random symmetric matrix of dimension NxN
    """
    A = np.random.rand(N, N)
    A = (A + A.conj().T) / 2.  # Symmetrize A
    
    return A

# %%
def to_a_a_dagger_basis(M):
    """
    Transforms a given Hamiltonian matrix from qp basis into a_dagger a basis.
    The convention is that the old matrix is in the order (q_1, q_2,....,q_n,p_1, p_2,....,p_n).
    In the new basis the order is (a_1, a_2, ...., a_n, a_1^dagger, a_2^dagger, ...., a_n^dagger).
    If you use the ordering (q_1, p_1, q_2, p_2,..., q_n, p_n) use 
    from thewalrus.symplectic import xpxp_to_xxpp, xxpp_to_xpxp to change between the ordering
    
    Parameters:
    M (numpy.ndarray): The Hamiltonian matrix of dimension (2N x 2N).
    
    Returns:
    numpy.ndarray: The transformed Hamiltonian matrix in the new basis.
    """
    # Get the shape of the Hamiltonian matrix and divide by 2, 
    # since for N particles the matrix has dimension 2N x 2N
    N = M.shape[0] // 2
    
    # Create Basis Change Matrix (Omega)
    identity_n = np.eye(N)
    Omega = 1/np.sqrt(2) * np.block(
        [[identity_n, identity_n], 
         [-1j * identity_n, 1j * identity_n]]
    )
    
    # Transform to Hamiltonian Matrix in new basis
    M_new_basis = Omega.conj().T @ M @ Omega
    
    # Return the real part if close to zero imaginary part
    return np.real_if_close(M_new_basis)

def to_quadrature_basis(h):
    """
    Transforms a given Hamiltonian matrix from a_dagger a basis into qp basis.
    The convention is that the old matrix is in the order (a_1, a_2, ...., a_n, a_1^dagger, a_2^dagger, ...., a_n^dagger).
    In the new basis the order is (q_1, q_2,....,q_n, (p_1, p_2,....,p_n).
    If you use the ordering (q_1, p_1, q_2, p_2,..., q_n, p_n) you can change the ordering afterwards via
    the functions xpxp_to_xxpp and xxpp_to_xpxp from the package thethewalrus.symplectic.
    
    Input Parameters:
    h: The Hamiltonian matrix in adagger adagger a a basis
    
    Return:
    h_new_basis: The Hamiltonian matrix in qqpp basis rounded to "digits" and real if its close
    """
    #get the shape of the hamiltonian matrix and dividde by 2, since for N particles the matrix has dimension 2N x 2N
    N = h.shape[0] // 2
    
    #create Basis Change Matrix
    identity_n = np.eye(N)
    Omega = 1/np.sqrt(2) * np.block( [[identity_n, identity_n], [-1j*identity_n, 1j*identity_n]])
    
    #Transform to Hamiltonian Matrix in new basis
    H_quadrature  = Omega @ h @ Omega.conj().T
    
    return np.real_if_close(H_quadrature)

def to_majorana_basis(H):
    """
    Converts a Hamiltonian matrix written in fermionic ladder operators to the Majorana basis.
    
    Parameters:
    H (numpy.ndarray): The Hamiltonian matrix of dimension (2N x 2N).
    The convention of the ordering used in the collection of the cration and annihilation operators.
    Here, this aligns with the Nuesseler paper (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.012415)
    In this convention the Hamiltonian matrix has the block form ((A -B^*), (B, -A^*)) where A is an Hermitian 
    and B is an antisymmetric N x N matrix.

    Returns:
    numpy.ndarray: The Hamiltonian matrix in Majorana basis, which is real by construction.
    """
    # Get the shape of the Hamiltonian matrix and divide by 2, 
    # since for N particles the matrix has dimension 2N x 2N
    N = H.shape[0] // 2
    
    # Create Basis Change Matrix (Omega)
    identity_n = np.eye(N)

    Omega = 1/np.sqrt(2) * np.block(
        [[identity_n, identity_n], 
            [ -1j*identity_n, 1j*identity_n]]
    )
    # Transform to Hamiltonian Matrix in Majorana basis
    H_majorana = Omega @ H @ Omega.conj().T
    
    # Return the real part if close to zero imaginary part
    return np.real_if_close(-1j*H_majorana)

# %%
def diagonalize_and_decompose_bosonic_hamiltonian(H, basis = "a_adagger"): 
    """
    Diagonalizes a bosonic Hamiltonian using the Williamson decomposition and Bloch-Messiah decomposition.
    Parameters:
    H (numpy.ndarray): The Hamiltonian matrix to be diagonalized. The ordering should be (q_1, q_2,....,q_n, p_1, p_2,....,p_n).
    In the ladder operation case it should be  (a_1, a_2, ...., a_n, a_1^dagger, a_2^dagger, ...., a_n^dagger).
    basis (str, optional): The basis in which to return the decomposed matrices. 
                            Can be either "a_adagger" or "q_p". Default is "a_adagger".
    Returns:
    tuple: A tuple containing three matrices (O, D, Q) in the specified basis:
        - O: Orthogonal matrix from the Bloch-Messiah decomposition.
        - D: Diagonal matrix from the Bloch-Messiah decomposition.
        - Q: Symplectic matrix from the Bloch-Messiah decomposition.
    Raises:
    ValueError: If the provided basis is not "a_adagger" or "q_p".
    """

    if basis == "a_adagger":
        H = to_quadrature_basis(H) #change to q_p basis
        _, S = williamson(H) #find the diagonal matrix and symplectic matrix S, s.t S*D*S.T = H in (q-1, q_1.. q_n, p_1, p_2... p_n) ordering
        O_bloch_messiah, D_bloch_messiah, Q_bloch_messiah = blochmessiah(S) # bloch messiah decomposition
        return to_a_a_dagger_basis(O_bloch_messiah), to_a_a_dagger_basis(D_bloch_messiah), to_a_a_dagger_basis(Q_bloch_messiah)
    
    elif basis == "q_p":
        _, S = williamson(H) #find the diagonal matrix and symplectic matrix S, s.t S*D*S.T = H in (q-1, q_1.. q_n, p_1, p_2... p_n) ordering
        O_bloch_messiah, D_bloch_messiah, Q_bloch_messiah = blochmessiah(S) # bloch messiah decomposition
        return O_bloch_messiah, D_bloch_messiah, Q_bloch_messiah
    else:
        raise ValueError("basis should be either a_adagger or q_p")

def diagonalize_fermionic_hamiltonian_in_majorana_basis(H):
    """
    Diagonalizes a fermionic Hamiltonian matrix in Majorana basis. The resulting matrix will be 
    block diagonal, ensuring positive upper diagonal elements. Additionally, the first half of 
    the blocks and the second half of the blocks are each sorted separately in descending order.
    
    Parameters:
    H: Hamiltonian Matrix in Majorana Basis
    
    Returns:
    h_d: Block diagonalized Hamiltonian matrix with ordered blocks
    O: Orthogonal matrix that block diagonalizes H such that h_d = O H O.T
    """
    
    assert np.allclose(H, -H.T, rtol=1e-04, atol=1e-10), "Input Matrix is not skew symmetric!"
    
    # Get the shape and define N
    N = H.shape[0] // 2  # The total matrix is 2N x 2N, meaning N blocks per half
    
    # Perform Schur decomposition
    h_d, O = scipy.linalg.schur(H, output='real')
    
    #print(np.round(h_d, 8))
    # Construct permutation matrix S to make upper diagonal elements positive
    S1 = np.array([[0, 1], [1, 0]])
    S2 = np.array([[1, 0], [0, 1]])
    S_blocks = []
    block_values = []  # Stores the values of the upper-diagonal elements for sorting
    
    for i in range(N):  # Iterate over N blocks
        value = h_d[2 * i, 2 * i + 1]
        if value < 0:
            S_blocks.append(S1)
            block_values.append(-value)  # Store absolute value
        else:
            S_blocks.append(S2)
            block_values.append(value)
    
    S = scipy.linalg.block_diag(*S_blocks)
    print(S.shape)
    # Update O and construct new h_d
    O = O @ S

    sorted_indices = np.argsort(block_values)[::-1]  # Sort in descending order
    #print(sorted_indices)
    
    # Create a permutation matrix P to reorder the blocks
    P = np.zeros((2*N, 2*N))  # Identity matrix
    for new_idx, old_idx in enumerate(sorted_indices):
        P[2 * old_idx +1 , 2 * new_idx + 1] = 1
        P[2 * old_idx, 2 * new_idx] = 1
   
    # Apply the permutation
    O = O @ P # Adjust O accordingly
    h_d = O.T @ H @ O

    return np.real_if_close(h_d), np.real_if_close(O)

def diagonalize_fermionic_hamiltonian_in_dirac__operators(H):
    """
    Diagonalizes a fermionic Hamiltonian matrix in Dirac operators.
    Parameters:
    H (numpy.ndarray): The Hamiltonian matrix of shape (2N, 2N) where N is the number of particles.
    The convention of the ordering used in the collection of the cration and annihilation operators.
    Here, the convention aligns with the Nuesseler paper (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.012415)
    In this convention the Hamiltonian matrix has the block form ((A -B^*), (B, -A^*)) where A is an Hermitian 
    and B is an antisymmetric N x N matrix.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: The diagonalized Hamiltonian matrix in ladder operators.
        - numpy.ndarray: The unitary matrix that diagonalizes the Hamiltonian.
    """
        
    #get the shape of the hamiltonian matrix and dividde by 2, since for N particles the matrix has dimension 2N x 2N
    N = H.shape[0] // 2
    
    #Initialize and create the Basis Change matrix from qpqp tp qqpp ordering
    F =  np.zeros((2*N, 2*N), dtype=int)
    for i in range(N):
        F[i, 2*i] = 1 
        F[2*N-1 - i, 2*N-1 - 2*i] = 1
        
    #define Omega for basis change back to a_dagger_a or a a_dagger
    identity_n = np.eye(N)

    Omega = 1/np.sqrt(2) * np.block(
        [[identity_n, identity_n], 
            [ -1j*identity_n, 1j * identity_n]]
    )
    
    # Change Hamiltonian to Majorana Basis
    H_temp = to_majorana_basis(H)
    #H_temp = np.real(-1j*H_temp) #as we get ih from going to majorana basis
    
    #block diagonalize the Hamiltonian Matrix in Majorana Basis
    H_temp, O = diagonalize_fermionic_hamiltonian_in_majorana_basis(H_temp)
    #print(np.diag(H_temp, k=1), np.diag(H_temp, k=-1))
    
    # Bring H back to the ladder operators
    H_temp = F @ H_temp @ F.conj().T
    H_temp = 1j* ((Omega.conj().T) @ H_temp @ Omega) 

    #Create the Unitary that Diagonalizes the Hamiltonian Matrix
    U_f = (Omega.conj().T) @ O @ F.T @ Omega
    
    return np.real(H_temp), np.real_if_close(U_f) 

def diagonalize_fermionic_hamiltonian(H, convention = "nuesseler", tol = 1e-10):
    """Diagonalizes a fermionic Hamiltonian matrix according to different conventions.
    This function diagonalizes a 2N x 2N fermionic Hamiltonian matrix using different conventions
    for the ordering of creation/annihilation operators. The supported conventions are:
    - "nuesseler": Nuesseler convention where operators are ordered as vec(a) =(a_1,...,a_N,a_1^†,...,a_N^†) and vec(a)^† H vec(a) = hat(H)
    - "surace": Surace convention with  operator ordering (a_1^†,...,a_N^†, a_1,...,a_N) and vec(a)^† H vec(a) = hat(H)
    - "serafini": Serafini convention where operators are ordered as (a_1,a_1^†,...,a_N,a_N^†) and vec(a)^T H vec(a) = hat(H)
    Parameters
    ----------
    H : numpy.ndarray
        2N x 2N complex Hamiltonian matrix to be diagonalized
    convention : str, optional
        Convention to use for operator ordering. Must be one of "nuesseler", "surace", or 
        "serafini". Default is "nuesseler".
    tol : float, optional
        Tolerance for numerical comparisons. Default is 1e-10.
    Returns
    -------
    H_diag : numpy.ndarray
        The diagonalized Hamiltonian matrix
    T : numpy.ndarray
        The transformation matrix that diagonalizes H, such that H_diag = T^† H T
    Raises
    ------
    AssertionError
        If the input matrix H does not satisfy the required form for the chosen convention
    Notes
    -----
    For each convention, the function first checks if the matrix is already in block-diagonal 
    form. If so, it uses a simpler diagonalization procedure. Otherwise, it transforms the 
    matrix appropriately before diagonalization.
    """
    
    N = H.shape[0] // 2

    if convention == "nuesseler":
        assert (np.allclose(H[:N, :N], -H[N:, N:].conj(), atol=tol) and 
        np.allclose(H[:N, N:], -H[N:, :N].conj(), atol=tol)
       ), "Matrix does not have the right form for the Nuesseler convention!"
        
        if(np.allclose(H[:N, N:], np.zeros((N, N)), atol=tol) and 
        np.allclose(H[N:, :N], np.zeros((N, N)), atol=tol)
        ):
            print("Hamiltonian is particle preservering! Only need a passive transformation!") 
            eigvals, eigvecs = np.linalg.eigh(H[:N, :N])
            negative_count = np.sum(eigvals < 0)
            eigvecs = eigvecs[:, ::-1]
            T = np.block([[eigvecs, np.zeros((N, N))], [np.zeros((N, N)), eigvecs.conj()]])
            H_diag = T.conj().T @ H @ T

            return H_diag, T, negative_count
        
        else:
            return diagonalize_fermionic_hamiltonian_in_dirac__operators(H)
    
    elif convention == "surace":

        if(np.allclose(H[:N, N:], np.zeros((N, N)), atol=tol) and 
        np.allclose(H[N:, :N], np.zeros((N, N)), atol=tol)
        ):
            print("Hamiltonian is particle preservering! Only need a passive transformation!") 
            eigvals, eigvecs = np.linalg.eigh(H[:N, :N])
            negative_count = np.sum(eigvals < 0)
            T = np.block([[eigvecs, np.zeros((N, N))], [np.zeros((N, N)), eigvecs.conj()]])
            H_diag = T.conj().T @ H @ T

            return H_diag, T, negative_count
        
        else:
            identity_n = np.eye(N)
            zeros = np.zeros((N, N))

            F =  np.block(
                [[zeros, identity_n], 
                    [ identity_n, zeros]]
            )

            H_temp = F.T @ H @ F

            H_diag, T = diagonalize_fermionic_hamiltonian_in_dirac__operators(H_temp)

            H_diag = F @ H_diag @ F.T
            T = F @ T @ F.T

            return H_diag, T
    
    elif convention == "serafini":

        assert (np.allclose(H[:N, :N], -H[N:, N:].conj(), atol=tol) and 
        np.allclose(H[:N, N:], -H[N:, :N].conj(), atol=tol)
        ), "Matrix does not have the right form for the Surace convention!"

        if(np.allclose(H[:N, :N], np.zeros((N, N)), atol=tol) and 
        np.allclose(H[N:, N:], np.zeros((N, N)), atol=tol)
        ):
            print("Hamiltonian is particle preservering! Only need a passive transformation!") 
            eigvals, eigvecs = np.linalg.eigh(H[:N, N:])
            negative_count = np.sum(eigvals < 0)
            T = np.block([[eigvecs.conj(), np.zeros((N, N))], [np.zeros((N, N)), eigvecs]])
            H_diag = T.T @ H @ T

            return H_diag, T, negative_count
        
        else:
            identity_n = np.eye(N)
            zeros = np.zeros((N, N))

            F =  np.block(
                [[zeros, identity_n], 
                    [ identity_n, zeros]]
            )

            H_temp = F @ H

            H_diag, T = diagonalize_fermionic_hamiltonian_in_dirac__operators(H_temp)

            H_diag = F @ H_diag 
            T = F @ T.conj()

            return H_diag, T

# %%
def check_for_unique_lambdas(lambdas, tol= 5e-10):
    """
    Checks for consecutive elements of an array that are equal within a given tolerance and groups them.
    Parameters
    ----------
    lambdas : array-like
        The array of eigenvalue-like elements to process.
    tol : float, optional
        The tolerance used to determine whether two values are considered equal. Default is 5e-10.
    Returns
    -------
    unique_lambdas : numpy.ndarray
        An array of values from the input that are considered unique according to the specified tolerance.
    indices : numpy.ndarray
        The corresponding indices of the first occurrence of each unique value in the input.
    counts : numpy.ndarray
        The count of how many consecutive values in the input matched each unique value within the tolerance.
    """
    unique_lambdas = [lambdas[0]]
    indices = [0]
    counts = [1]
    for i in range(1, len(lambdas)):
        if abs(lambdas[i] - unique_lambdas[-1]) < tol:
            counts[-1] += 1
        else:
            unique_lambdas.append(lambdas[i])
            indices.append(i)
            counts.append(1)

    # Convert to numpy arrays
    unique_lambdas = np.array(unique_lambdas)
    indices = np.array(indices)
    counts = np.array(counts)
    
    return unique_lambdas, indices, counts

def create_S_i_for_subspace(eigenvalue, degeneracy_of_eigenvalue, eigenvectors, P_dagger, epsilon = 1e-10):
    """
    This is a helper function for the function bring_skew_symmetric_into_canonical_form that
    brings an antisymmetric matrix to its canonical form. The canonical form has blocks with
    pairwise real numbers on the diagonal and 0 at the end.
    This function creates S_i for each subspace of degenerate eigenvalues. It creates orthogonal
    vectors belonging to the same eigenvalue as the given (eigenvector, eigenvalue) pair and
    returns the part of S, which is the matrix such that S.T P S is in canonical form for the
    given subspace.
    Parameters
    ----------
    eigenvalue : float
        The eigenvalue for the subspace.
    degeneracy_of_eigenvalue : int
        The degeneracy of that eigenvalue (i.e., how many times the eigenvalue occurs).
    eigenvectors : np.ndarray
        The eigenvectors found by eigh for that eigenvalue.
    P_dagger : np.ndarray
        The conjugate transpose of the matrix P that is transformed to the canonical form.
    epsilon : float, optional
        Tolerance parameter to decide if something is close enough to zero to discard.
        Default is 1e-10.
    Returns
    -------
    np.ndarray
        The matrix S_i such that S_i.T P_i S_i is in the canonical form (i.e., S for the
        subspace of the eigenvalue).
    """
    
    #initialize a counter to count how many vectors we already have. We need in the end as many vectors as the degeneracy
    counter = 0
    # initialize the by construction unitary matrix S_i that brings the subspace in its canonical form
    S_i = np.empty((eigenvectors.shape[0], degeneracy_of_eigenvalue), dtype = np.complex128)
    
    #loop over the degeneracy_of_eigenvalue
    for i in range(degeneracy_of_eigenvalue):
        if counter == degeneracy_of_eigenvalue: #break if we already have enough vectors
            break
            
        #get the ith eigenvector corresponding to the eigenvalue
        eigenvector = eigenvectors[:,i] 

        #project everything that is not orthogonal out 
        subtract_vector  = np.zeros(eigenvectors.shape[0])
        for k in range(counter):
            subtract_vector += np.vdot(S_i[:,k], eigenvector) * S_i[:, k] #vdot since complex
        eigenvector = eigenvector - subtract_vector

        #calculate the norm of the projected eigenvector
        norm = np.linalg.norm(eigenvector)
        # Check if the norm is still bigger than the tolerance
        if norm > epsilon:  
            eigenvector = (1 / norm) * eigenvector  #normalize the eigenvector
            w_i = (1 / (np.sqrt(np.abs(eigenvalue)))) * P_dagger @ eigenvector.conj() #create W that is orthogonal to
            #the eigenvector. 

            #fill the Matrix s_i 
            S_i[:,2*i] = eigenvector
            S_i[:,2*i + 1] = w_i
            counter = counter + 2 # add the number of addred vectors to the counter

    return S_i

def bring_skew_symmetric_into_canonical_form_high_to_low(P, tol=5e-10):
    """
    Bring a skew-symmetric matrix P into its canonical form.
    This function computes a unitary transformation S such that S.T @ P @ S is in
    a canonical, block-diagonal form. It approximately identifies zero eigenvalues
    and handles numerical precision issues by means of a tolerance.
    Parameters:
        P (numpy.ndarray):
            A skew-symmetric (Hermitian in the imaginary sense) matrix for which
            the canonical form is to be found.
        tol (float, optional):
            Tolerance for considering eigenvalues as zero or identical. Defaults
            to 5e-10.
    Returns:
        tuple:
            A tuple (P_canonical, S), where:
            - P_canonical (numpy.ndarray): The matrix P in its canonical form.
            - S (numpy.ndarray): The unitary matrix that transforms P into its
              canonical form via S.T @ P @ S.
    """
    
    #get the shape of the hamiltonian matrix
    N = P.shape[0]
    #define P_dagger @ P
    P_dagger = P.conj().T
    Q = P_dagger @ P
    
    #get eigenvalues and eigenvectors
    lambda_i, v_i = np.linalg.eigh(Q)
    
    #get unique lambdas and their degeneracies
    unique_lambdas, index_lambdas, degeneracies = check_for_unique_lambdas(lambda_i, tol = tol)
    
    #create the unitary matrix that brings P into its canonical form
    S = np.empty((N, N), dtype = np.complex128) #np.complex128 for precision
    index_counter = 0
    #First v_i that have eigenvalue 0
    index_eigenvalue_0 = np.where(np.abs(lambda_i < tol))[0] 
    degeneracy_0 = len(index_eigenvalue_0)
    S[:, index_counter: index_counter + degeneracy_0] = v_i[:, index_eigenvalue_0]
    #increase the index_counter
    index_counter = index_counter +  degeneracy_0
    
    # Now the rest
    for i in range(len(unique_lambdas)):
        unique_lambda = unique_lambdas[i]
        index = index_lambdas[i]
        degeneracy = degeneracies[i]
        
        if np.abs(unique_lambda) >= tol:
            S_i = create_S_i_for_subspace(unique_lambda,degeneracy , v_i[:, index: index+degeneracy], P_dagger)
            S[:, index_counter: index_counter + degeneracy] = S_i
            index_counter = index_counter + degeneracy
    
    return S.T @ P @ S, S

def bring_skew_symmetric_into_canonical_form_low_to_high(P, tol = 5e-10):
    """
    Brings a skew-symmetric matrix P into its canonical form (from low to high eigenvalues)
    by constructing a suitable transformation matrix S. The function internally computes
    eigenvalues and eigenvectors of P†P to determine the subspace structure.
    Parameters
    ----------
    P : numpy.ndarray
        The skew-symmetric matrix to be transformed.
    tol : float, optional
        Numerical tolerance for identifying zero eigenvalues and determining unique eigenvalues.
        Defaults to 5e-10.
    Returns
    -------
    tuple
        A tuple (canonical_form, S) where:
        - canonical_form (numpy.ndarray): The transformed form of P in the canonical basis.
        - S (numpy.ndarray): The unitary matrix used to transform P into its canonical form.
    Notes
    -----
    - The function orders the eigenvalues in ascending order.
    - Eigenvalues smaller than or equal to the provided tolerance are considered zero.
    - The return value includes both the canonical form of the original matrix and the
        transformation matrix S, such that (S.T @ P @ S) yields the canonical form.
    """
    
    #get the shape of the hamiltonian matrix
    N = P.shape[0]
    #define P_dagger @ P
    P_dagger = P.conj().T
    Q = P_dagger @ P
    
    #get eigenvalues and eigenvectors
    lambda_i, v_i = np.linalg.eigh(Q)

    #get unique lambdas and their degeneracies
    unique_lambdas, index_lambdas, degeneracies = check_for_unique_lambdas(lambda_i, tol = tol)
    #reverse their order so that we get the diagonal of U in ascending order later
    unique_lambdas = unique_lambdas[::-1]
    index_lambdas = index_lambdas[::-1]
    degeneracies = degeneracies[::-1]
    
    #create the unitary matrix that brings P into its canonical form
    S = np.empty((N, N), dtype = np.complex128)
    index_counter = 0
    for i in range(len(unique_lambdas)):
        unique_lambda = unique_lambdas[i]
        index = index_lambdas[i]
        degeneracy = degeneracies[i]
        
        if np.abs(unique_lambda) >= tol: #check if the eigenvalue is not 0. If true go to the helper function and create S_i for subspace
            S_i = create_S_i_for_subspace(unique_lambda,degeneracy , v_i[:, index: index+degeneracy], P_dagger)
            S[:, index_counter: index_counter + degeneracy] = S_i #add the subspace to S
            index_counter = index_counter + degeneracy
                   
    #rest with v_i that have eigenvalue 0
    #get the index and degeneracy of the eigenvalue 0 
    index_eigenvalue_0 = np.where(np.abs(lambda_i) < tol) [0] 
    S[:, index_counter:] = v_i[:, index_eigenvalue_0]
    
    return S.T @ P @ S, S

def bloch_messiah_high_to_low(W, tol = 5e-10):
    """
    Perform the Bloch-Messiah decomposition on the given 2N×2N matrix W, returning
    the transformed blocks C, Q, and D. The function internally uses SVD and a skew-
    symmetric canonical form routine to ensure a proper decomposition under floating-
    point precision constraints.
    Parameters
    ----------
    W : numpy.ndarray
        A 2N×2N matrix containing the blocks for which the Bloch-Messiah decomposition
        is to be performed.
    tol : float, optional
        A numerical tolerance used for singular values and to account for floating-point
        precision errors when determining zero values. Defaults to 5e-10.
    Returns
    -------
    C : numpy.ndarray
        A 2N×2N matrix containing one of the transformation blocks of the decomposition.
    Q : numpy.ndarray
        A 2N×2N matrix representing the central canonical form block after the
        transformations.
    D : numpy.ndarray
        A 2N×2N matrix containing the complementary transformation block of the
        decomposition.
    Notes
    -----
    - The function checks singular values against the tolerance to identify and
      handle near-zero modes appropriately.
    - SVD is performed on the U block of W, and a specialized method is used to
      bring the V block of W to a canonical skew-symmetric form when possible.
    - This function is particularly useful in the context of Gaussian-state algebra
      and Hamiltonian diagonalization.
    """
    
    #calculate tolerance factor for if close. That is given as a multiple of 2,2204460492503131e-16 
    real_tolerance = tol/ (2.22e-16)

    #get shape of the blocks of the input matrix
    N = W.shape[0] // 2
    
    #extract the NxN blocks U and V from the unitary matrix that diagonalizes the hamiltonian
    U = W[: N, :N]
    V = W[:N, N: 2 *N]
    
    U = U.astype(np.complex128) #just to make sure the precision is good enough in the end
    V = V.astype(np.complex128)
    
    #SVD on U block: 
    A, s, B = scipy.linalg.svd(U)
    A = A.conj().T #to get the same SVD as we have in our proof.
    
    #check how many 0 are in the singular values
    number_of_zeros = np.count_nonzero(s <= tol)

    #extract the block AVB.T. By construction the block AUB_Dagger is diagonal 
    A_V_B_transpose = A @ V @ B.T
    
    if number_of_zeros != 0:
        #Get the Part of AVB.T that is not antisymmetric. This corresponds to u_i = 0 in the diagonal AUB_dagger part
        #Since UU_dagger + VV_dagger = 1 = A_daggerUU_daggerA + A VV_dagger A_dagger  we now that if u_i = 1 v_i has to be 0 
        P_m_x_m = A_V_B_transpose[N - number_of_zeros: N, N - number_of_zeros : N]
        #Perform an SVD On that part
        F_n, _, G_n = scipy.linalg.svd(P_m_x_m)
        #to be consistent with our notation
        F_n = F_n.conj().T

        #build the new blocks that make sure that P has the correct form
        K = scipy.linalg.block_diag(np.eye(N - number_of_zeros, N - number_of_zeros), F_n)
        A = K @ A
        L = scipy.linalg.block_diag(np.eye(N - number_of_zeros, N - number_of_zeros), G_n.conj())
        B = L @ B
        #recalculate  A_V_B_transpose with the new matrices
        A_V_B_transpose = A @ V @ B.T
  
        # extract the block above the 1 to feed it into the canonical form algorithm
        anti_symmetric_part_of_A_V_B_transpose = A_V_B_transpose[0:N - number_of_zeros, 0: N - number_of_zeros] 
        _, R = bring_skew_symmetric_into_canonical_form_high_to_low(anti_symmetric_part_of_A_V_B_transpose, tol = tol)

        #add the identity block to R
        R = scipy.linalg.block_diag(R, np.eye(number_of_zeros)) 

        #Update the A and B blocks. R will leave AUB_dagger unchanged but brings AVB.T in canonical form
        A = R.T @ A
        B = R.T @ B #tramspose since R.T @ A_V_B_transpose @ R = canonical form
        A_block = scipy.linalg.block_diag(A, A.conj())
        B_block = scipy.linalg.block_diag(B.conj().T, B.T)

        #re-calculate A @ W @ B and construc the bloch messiah matrices
        Q = A_block @ W @ B_block
        C = A_block.conj().T
        D = B_block.conj().T
        
        #floating point precision which is roughly 2e-16 typically 
        return np.real_if_close(C), np.real_if_close(Q), np.real_if_close(D)     
    
    else:
        # extract the block above the 1 to feed it into the canonical form algorithm
        _, R = bring_skew_symmetric_into_canonical_form_high_to_low(A_V_B_transpose, tol = tol)

        #Update the A and B blocks. R will leave AUB_dagger unchanged but brings AVB.T in canonical form
        A = R.T @ A 
        B = R.T @ B #tramspose since R.T @ A_V_B_transpose @ R = canonical form
        A_block = scipy.linalg.block_diag(A, A.conj())
        B_block = scipy.linalg.block_diag(B.conj().T, B.T)

        #re-calculate A @ W @ B and construc the bloch messiah matrices
        Q = A_block @ W @ B_block
        C = A_block.conj().T
        D = B_block.conj().T
        
        return np.real_if_close(C, tol = real_tolerance), np.real_if_close(Q, tol = real_tolerance), np.real_if_close(D, tol = real_tolerance)  

def bloch_messiah_low_to_high(W, tol = 5e-10):
    """
    Perform the Bloch-Messiah decomposition on a given unitary matrix W.
    Parameters:
    W (numpy.ndarray): A 2N x 2N unitary matrix that diagonalizes the Hamiltonian.
    tol (float, optional): Tolerance for numerical precision. Default is 5e-10.
    Returns:
    tuple: A tuple containing three numpy.ndarrays (C, Q, D) which are the Bloch-Messiah matrices.
        - C (numpy.ndarray): The left unitary matrix in the decomposition.
        - Q (numpy.ndarray): The diagonal matrix in the decomposition.
        - D (numpy.ndarray): The right unitary matrix in the decomposition.
    """

    #calculate tolerance factor for if close. That is given as a multiple of 2,2204460492503131e-16 
    real_tolerance = tol/ 2.22e-16

    #get shape of the blocks of the input matrix
    N = W.shape[0] // 2
    
    #extract the NxN blocks U and V from the unitary matrix that diagonalizes the hamiltonian
    U = W[: N, :N]
    V = W[:N, N: 2 *N]
    
    U = U.astype(np.complex128) #make sure to have the np.complex128 for floating point precision np.complex64 was not precise                           #
    V = V.astype(np.complex128) # enough
    
    #SVD on U block:
    A, s, B = scipy.linalg.svd(U)
    
    #reorder such that s is in increasing order instead of decreasing
    D = np.flip(np.eye(N), 1)
    A = A @ D
    B = D @ B
    s = s[::-1]
    
    #check how many 0 are in the singular values
    number_of_zeros = np.count_nonzero(s <= tol)
    
    #to get the same SVD as we have in our proof.
    A = A.conj().T 

    #extract the block AUB_dagger and AVB.T. By construction AUB_Dagger is diagonal 
    A_V_B_transpose = A @ V @ B.T
    
    #Get the Part of AVB.T that is not antisymmetric. This corresponds to u_i = 0 in the diagonal AUB_dagger part
    #Since UU_dagger + VV_dagger = 1 = A_daggerUU_daggerA + A VV_dagger A_dagger  we now that if u_i = 1 v_i has to be 0 
    if number_of_zeros != 0:
        P_m_x_m = A_V_B_transpose[: number_of_zeros, :number_of_zeros]
        #Perform an SVD On that part
        F_n, _, G_n = scipy.linalg.svd(P_m_x_m)
        #to be consistent with our notation
        F_n = F_n.conj().T
        
        #build the new blocks that make sure that P has the correct form
        K = scipy.linalg.block_diag( F_n, np.eye(N - number_of_zeros, N - number_of_zeros))
        A = K @ A
        L = scipy.linalg.block_diag(G_n.conj(), np.eye(N - number_of_zeros, N - number_of_zeros))
        B = L @ B
        #recalculate A_U_B_dagger and A_V_B_transpose with the new matrices
        A_V_B_transpose = A @ V @ B.T

        # extract the block above the 1 to feed it into the canonical form algorithm
        anti_symmetric_part_of_A_V_B_transpose = A_V_B_transpose[number_of_zeros :N, number_of_zeros: N]
        _, R = bring_skew_symmetric_into_canonical_form_low_to_high(anti_symmetric_part_of_A_V_B_transpose, tol = tol)
        
        #add the identity block to R
        R = scipy.linalg.block_diag(np.eye(number_of_zeros),R) 

        #Update the A and B blocks. R will leave AUB_dagger unchanged but brings AVB.T in canonical form
        A = R.T @ A
        B = R.T @ B #tramspose since R.T @ A_V_B_transpose @ R = canonical form
        A_block = scipy.linalg.block_diag(A, A.conj())
        B_block = scipy.linalg.block_diag(B.conj().T, B.T)

        #re-calculate A @ W @ B and construct the bloch messiah matrices
        Q = A_block @ W @ B_block
        C = A_block.conj().T
        D = B_block.conj().T
        
        return np.real_if_close(C), np.real_if_close(Q), np.real_if_close(D)  
    
    else:
        # extract the block above the 1 to feed it into the canonical form algorithm
        _, R = bring_skew_symmetric_into_canonical_form_low_to_high(A_V_B_transpose, tol = tol)

        #Update the A and B blocks. R will leave AUB_dagger unchanged but brings AVB.T in canonical form
        A = R.T @ A 
        B = R.T @ B #tramspose since R.T @ A_V_B_transpose @ R = canonical form
        A_block = scipy.linalg.block_diag(A, A.conj())
        B_block = scipy.linalg.block_diag(B.conj().T, B.T)

        #re-calculate A @ W @ B and construc the bloch messiah matrices
        Q = A_block @ W @ B_block
        C = A_block.conj().T
        D = B_block.conj().T
        
        return np.real_if_close(C, tol = real_tolerance), np.real_if_close(Q, tol = real_tolerance), np.real_if_close(D, tol = real_tolerance)  

# %% #"max_trunc_err": 1e-10,  "svd_min": 1e-10
def apply_interferometer(psi, operator_list, inverse = False, trunc_params = { "chi_max": 150}, renormalize = False, print_enable = False):
    """
    Apply a sequence of interferometer operations to a given MPS (Matrix Product State).
    Parameters:
    -----------
    psi : MPS
        The Matrix Product State to which the interferometer operations will be applied.
    operator_list : list of tuples
        A list of tuples where each tuple contains the parameters for the interferometer operation.
        Each tuple should be of the form (left_site, right_site, theta, phi).
    inverse : bool, optional
        If True, apply the inverse of the interferometer operations. Default is False.
    trunc_params : dict, optional
        Parameters for truncation during compression. Default is {"max_trunc_err": 1e-8, "chi_max": 100, "svd_min": 1e-12}.
    print_enable : bool, optional
        If True, print information about the applied operators. Default is False.
    Returns:
    --------
    psi : MPS
        The Matrix Product State after applying the interferometer operations.
    Raises:
    -------
    ValueError
        If the site type of the MPS is not BosonSite, FermionSite, or SpinHalfSite.
    """

    #check for inverse operation
    if inverse == False:
        sign = 1
    elif inverse == True:
        sign = -1   

    if isinstance(psi.sites[0], site.BosonSite):

        a_dagger =  psi.sites[0].get_op('Bd') # get a_dagger operator for cutoff dimension M
        a =  psi.sites[0].get_op('B') # get a operator for cutoff dimension M
        N = psi.sites[0].get_op('N') # get the number operator
            
        #pre calculate a tensor a_dagger and a_dagger tensor a for the beam splitter operator
        a_adagger = site.kron(a,a_dagger)
        adagger_a = site.kron(a_dagger,a)

        for i, op in enumerate(operator_list):

            left_site = op[0] #left site of the beam splitter
            theta = op[2] #beam splitter angle
            phi = op[3] #phase shifter angle#

            if print_enable:
                print(f"Applying operator number {i + 1} of {len(operator_list)} \n The Paramters are theta: {sign*theta}, phi: {sign*phi}")

            #create the Beam Splitter operator
            Beam_splitter = npc.expm(sign*theta*(a_adagger - adagger_a)) # create the beam splitter operator
            Beam_splitter = Beam_splitter.split_legs().itranspose([0,2,1,3])

            #create the Phase Shifter operator
            Phase_shifter = npc.expm(1j*sign*phi*N) # create the phase shifter operator
            
            if inverse == False:
                #apply the operators to the MPS in the order Phase shifter -> Beam splitter
                psi.apply_local_op(left_site, Phase_shifter, renormalize=renormalize)
                psi.apply_local_op(left_site, Beam_splitter, renormalize=renormalize)
                #compression for Bosons important as bond dimensino can get very large. Prevent that with this compression
                psi.compress(options = {'compression_method': 'SVD', 'trunc_params': trunc_params}) 
            elif inverse == True:
                #apply the operators to the MPS in the order Beam splitter -> Phase shifter
                psi.apply_local_op(left_site, Beam_splitter, renormalize=renormalize)
                psi.apply_local_op(left_site, Phase_shifter, renormalize=renormalize)
                psi.compress(options = {'compression_method': 'SVD', 'trunc_params': trunc_params}) 

    elif isinstance(psi.sites[0], site.FermionSite) or isinstance(psi.sites[0], site.SpinHalfSite):

        if isinstance(psi.sites[0], site.FermionSite):
            charge_leg = psi.sites[0].leg #get leg charge for the operator
            sigma_plus = npc.Array.from_ndarray(np.array([[0,0], [1,0]], dtype=np.float64), legcharges = [charge_leg, charge_leg.conj()] , labels = ['p', 'p*'])
            sigma_minus = npc.Array.from_ndarray(np.array([[0,1], [0,0]], dtype=np.float64), legcharges = [charge_leg, charge_leg.conj()] , labels = ['p', 'p*'])
            sigma_plus_times_sigma_minus = npc.tensordot(sigma_plus, sigma_minus, axes = 1)
            
            #pre calculate a tensor a_dagger and a_dagger tensor a for the beam splitter operator
            sigma_plus_sigma_minus = site.kron(sigma_plus,sigma_minus)
            sigma_minus_sigma_plus = site.kron(sigma_minus,sigma_plus)
        
        elif isinstance(psi.sites[0], site.SpinHalfSite):
            sigma_plus =  psi.sites[0].get_op('Sp') # create s_+ operator
            sigma_minus = psi.sites[0].get_op('Sm') # create s_- operator
            sigma_plus_times_sigma_minus = npc.tensordot(sigma_plus, sigma_minus, axes = 1)

            #pre calculate a tensor a_dagger and a_dagger tensor a for the beam splitter operator
            sigma_plus_sigma_minus = site.kron(sigma_plus,sigma_minus)
            sigma_minus_sigma_plus = site.kron(sigma_minus,sigma_plus)

        else:
            raise ValueError("Site is not a FermionSite or SpinHalfSite and is not valid!")

        for op in operator_list:
            left_site = op[0] #left site of the beam splitter
            theta = op[2] #beam splitter angle
            phi = op[3] #phase shifter angle

            if print_enable:
                print(f"Applying operator number {i + 1} of {len(operator_list)} \n The Paramters are theta: {sign*theta}, phi: {sign*phi}")

            #create the Beam Splitter operator
            Beam_splitter = npc.expm(sign*theta*(sigma_minus_sigma_plus - sigma_plus_sigma_minus)) # create the beam splitter operator
            Beam_splitter = Beam_splitter.split_legs().itranspose([0,2,1,3])

            #create the Phase Shifter operator
            Phase_shifter = npc.expm(1j*sign*phi*sigma_plus_times_sigma_minus) # create the phase shifter operator

            #apply the operators to the MPS
            if inverse == False:
                psi.apply_local_op(left_site, Phase_shifter, renormalize=renormalize)
                psi.apply_local_op(left_site, Beam_splitter, renormalize=renormalize)
            elif inverse == True:
                psi.apply_local_op(left_site, Beam_splitter, renormalize=renormalize)
                psi.apply_local_op(left_site, Phase_shifter, renormalize=renormalize)
    else:
        raise ValueError("Site is not a valid site! Valid Site types are BosonSite, FermionSite and SpinHalfSite")
    
    return psi
    
def apply_squeezing_operation(psi, S):
    """
    Apply a squeezing operation to a given state `psi` using the squeezing matrix `S`.
    Parameters:
    -----------
    psi : MPS
        The many-body state to which the squeezing operation will be applied. The state should be an instance of 
        a Matrix Product State (MPS) with sites that are either BosonSite, FermionSite, or SpinHalfSite.
    S : numpy.ndarray
        The squeezing matrix of shape (2N, 2N) where N is the number of sites. This matrix contains the squeezing 
        parameters for the operation.
    Returns:
    --------
    psi : MPS
        The state after the squeezing operation has been applied.
    Raises:
    -------
    ValueError
        If the site type of `psi` is not BosonSite, FermionSite, or SpinHalfSite.
    """

    N = S.shape[0] // 2 #since matrix is 2N by 2N and we have N sites

    if isinstance(psi.sites[0], site.BosonSite):
        a_dagger =  psi.sites[0].get_op('Bd') # get a_dagger operator for cutoff dimension M
        a =  psi.sites[0].get_op('B') # get a operator for cutoff dimension M

        #create the Squeezing operator
        a_tensor_a = npc.tensordot(a, a, axes = 1)
        a_dagger_tensor_a_dagger = npc.tensordot(a_dagger, a_dagger, axes = 1)

        S  = S[:N, :2*N] #to have only the first N rows and 2N columns for N sites.
        cosh_m = np.diag(S) #cosh(m) is on the "main diagonal"
        sinh_m = np.diag(S, N) #sinh(m) is on the mth "diagonal"
        tanh_m = sinh_m/cosh_m #tanh to take care of signs as cosh is symmetric 
        z = 0.5*np.arctanh(tanh_m) #extract z. MAYBE 0.5 factor in front of arctanh is needed

        for i, z_i in enumerate(z):
            S = npc.expm(z_i*(a_tensor_a - a_dagger_tensor_a_dagger)) # create the one mode squeezing operator
            psi.apply_local_op(i, S, renormalize=False)

    elif isinstance(psi.sites[0], site.FermionSite) or isinstance(psi.sites[0], site.SpinHalfSite):
        S_gamma = S[:N, :N]
        S_mu = S[:N, N:2*N]

        gammas = np.diag(S_gamma)
        evens = np.arange(0, N, 2, dtype=int)
        odds = np.arange(1, N, 2, dtype = int)
        pattern = np.empty(evens.size + odds.size, dtype =int)
        pattern[0::2] = odds  # even indices
        pattern[1::2] = evens
        mus = []

        for i in range(S_mu.shape[0]):
            col_idx = pattern[i]
            mus.append(S_mu[i, col_idx])

        for i in range(0, N-1, 2):

            if np.isclose(gammas[i], 1):
                continue
            elif np.isclose(gammas[i], 0):
                apply_swap_operator(psi, i)
            else:
                z = np.arctan(mus[i] / gammas[i])
                print(z)
                apply_two_mode_squeezing_operator(psi, i, z)

        #while index_counter <  N:
            
        #    if np.isclose(S[index_counter][index_counter], 1): #if the element is 1, do nothing (act with identity)
        #        index_counter += 2

        #    elif np.isclose(S[index_counter][index_counter], 0): #if the element is 0, apply a swap operator
        #        apply_swap_operator(psi, index_counter)
        #        index_counter += 2
        #    else:
                #be careful with factor 0.5 in front 
        #        z = np.real(np.arctan( -S[index_counter][index_counter + N  + 1] / S[index_counter][index_counter])) #sin(z)/cos(z) and then arc tan to make sure that sings are correct
        #        print(z)
        #        apply_two_mode_squeezing_operator(psi, index_counter % N, z)  
        #        index_counter += 2 
    else:
        raise ValueError("Site is not a valid site! Valid Site types are BosonSite, FermionSite and SpinHalfSite")
    
    return psi

def apply_two_mode_squeezing_operator(psi, left_side, z):
    """
    Apply a two-mode squeezing operator to a given quantum state.
    Parameters:
    -----------
    psi : MPS
        The matrix product state (MPS) to which the two-mode squeezing operator will be applied.
    left_side : int
        The site index on the left side where the operator will be applied.
    z : complex
        The squeezing parameter, which is a complex number.
    Returns:
    --------
    MPS
        The modified matrix product state after applying the two-mode squeezing operator.
    Raises:
    -------
    ValueError
        If the site type is not FermionSite or SpinHalfSite.
    """

    if isinstance(psi.sites[0], site.FermionSite):         
        charge_leg = psi.sites[0].leg #get leg charge for the operator
        sigma_plus = npc.Array.from_ndarray(np.array([[0,0], [1,0]], dtype=np.float64), legcharges = [charge_leg, charge_leg.conj()] , labels = ['p', 'p*'])
        sigma_minus = npc.Array.from_ndarray(np.array([[0,1], [0,0]], dtype=np.float64), legcharges = [charge_leg, charge_leg.conj()] , labels = ['p', 'p*'])

        #pre calculate a tensor a_dagger and a_dagger tensor a for the beam splitter operator
        sigma_plus_sigma_plus = site.kron(sigma_plus,sigma_plus)
        sigma_minus_sigma_minus = site.kron(sigma_minus,sigma_minus)

    elif isinstance(psi.sites[0], site.SpinHalfSite):
        sigma_plus =  psi.sites[0].get_op('Sp') # create s_+ operator
        sigma_minus = psi.sites[0].get_op('Sm') # create s_- operator
         
        #pre calculate a tensor a_dagger and a_dagger tensor a for the beam splitter operator
        sigma_plus_sigma_plus = site.kron(sigma_plus,sigma_plus)
        sigma_minus_sigma_minus = site.kron(sigma_minus,sigma_minus)
    else:
        raise ValueError("Site is not a valid site! Valid Site types are FermionSite and SpinHalfSite")

    S = npc.expm(z*(sigma_plus_sigma_plus + sigma_minus_sigma_minus)) # create the squeezing operator
    S = S.split_legs().itranspose([0,2,1,3])
    psi.apply_local_op(left_side, S, renormalize=False)
    
    return psi

def apply_swap_operator(psi, swap_site):
    """
    Apply a swap operator to a given quantum state.
    This function applies a swap operator to a quantum state `psi` at a specified site `swap_site`.
    The swap operator is constructed differently depending on whether the site is a FermionSite or a SpinHalfSite.
    Parameters:
    -----------
    psi : MPS
        The quantum state to which the swap operator will be applied. It is expected to be an instance of an MPS (Matrix Product State).
    swap_site : int
        The site index at which the swap operator will be applied.
    Returns:
    --------
    MPS
        The quantum state after applying the swap operator.
    Raises:
    -------
    ValueError
        If the site type of `psi` is neither FermionSite nor SpinHalfSite.
    Notes:
    ------
    - For FermionSite, the swap operator is constructed using the charge leg of the site.
    - For SpinHalfSite, the swap operator is constructed using the `Sp` and `Sm` operators.
    - The `sigma_z` operator is applied to all sites from `swap_site-1` to 0.
    """

    if isinstance(psi.sites[0], site.FermionSite):
        charge_leg = psi.sites[0].leg #get leg charge for the operator
        sigma_plus_plus_sigma_minus= npc.Array.from_ndarray(np.array([[0,1], [1,0]], dtype=np.float64), legcharges = [charge_leg, charge_leg.conj()] , labels = ['p', 'p*'])
        sigma_z = npc.Array.from_ndarray(np.array([[1,0], [0,-1]], dtype=np.float64), legcharges = [charge_leg, charge_leg.conj() ] , labels = ['p', 'p*'])
    
    elif isinstance(psi.sites[0], site.SpinHalfSite):
        sigma_plus_plus_sigma_minus =  psi.sites[0].get_op('Sp') + psi.sites[0].get_op('Sm') #
        sigma_z = psi.sites[0].get_op('Sigmaz') # create s_z operator
    
    else:
        raise ValueError("Site is not a valid site! Valid Site types are FermionSite and SpinHalfSite")

    psi.apply_local_op(swap_site, sigma_plus_plus_sigma_minus, renormalize=False)
    for i in range(swap_site-1, -1, -1):
        psi.apply_local_op(i, sigma_z, renormalize=False)

    return psi

def apply_layer_of_phase_shifter(psi, D):
    """
    Apply a layer of phase shifters to the given state `psi` using the diagonal matrix `D`.
    Parameters:
    -----------
    psi : MPS
        The matrix product state (MPS) to which the phase shifters will be applied.
    D : np.ndarray
        A diagonal matrix whose angles will be used to create the phase shifters.
    Returns:
    --------
    psi : MPS
        The modified matrix product state after applying the phase shifters.
    Raises:
    -------
    ValueError
        If the site type of `psi` is not BosonSite, FermionSite, or SpinHalfSite.
    Notes:
    ------
    - For BosonSite, the phase shifter is created using the number operator `N`.
    - For FermionSite and SpinHalfSite, the phase shifter is created using the 
      product of the raising (`sigma_plus`) and lowering (`sigma_minus`) operators.
    """

    #get phis from the D matrix
    phis = np.angle(D)  

    if isinstance(psi.sites[0], site.BosonSite):
        N = psi.sites[0].get_op('N') # get the number operator
        #get phis from the D matrix
        phis = np.mod(np.angle(D), 2*np.pi)

        for i, phi in enumerate(phis):
            #create the Phase Shifter operator
            Phase_shifter = npc.expm(1j*phi*N) # create the phase shifter operator
            psi.apply_local_op(i, Phase_shifter, renormalize=False)
    
    elif isinstance(psi.sites[0], site.FermionSite) or isinstance(psi.sites[0], site.SpinHalfSite):

        if isinstance(psi.sites[0], site.FermionSite):

            charge_leg = psi.sites[0].leg #get leg charge for the operator
            sigma_plus = npc.Array.from_ndarray(np.array([[0,0], [1,0]], dtype=np.float64), legcharges = [charge_leg, charge_leg.conj()] , labels = ['p', 'p*'])
            sigma_minus = npc.Array.from_ndarray(np.array([[0,1], [0,0]], dtype=np.float64), legcharges = [charge_leg, charge_leg.conj()] , labels = ['p', 'p*'])
            sigma_plus_times_sigma_minus = npc.tensordot(sigma_plus, sigma_minus, axes = 1)
        
        elif isinstance(psi.sites[0], site.SpinHalfSite):
            sigma_plus =  psi.sites[0].get_op('Sp') # create s_+ operator
            sigma_minus = psi.sites[0].get_op('Sm') # create s_- operator
            sigma_plus_times_sigma_minus = npc.tensordot(sigma_plus, sigma_minus, axes = 1)

        
        for i, phi in enumerate(phis):
         Phase_shifter = npc.expm(1j*phi*sigma_plus_times_sigma_minus) # create the phase shifter operator
         psi.apply_local_op(i, Phase_shifter, renormalize=False)
    
    else:
        raise ValueError("Site is not a valid site! Valid Site types are BosonSite, FermionSite and SpinHalfSite")

    return psi 

def apply_passive_transformation(psi, U, decomposition_scheme = "rectangular", tol = 1e-11, trunc_params = { "chi_max": 150}, renormalize = False, print_enable = False):
    
    """
    Applies a passive linear transformation to a quantum state `psi` using a unitary matrix `U`.
    Parameters:
    -----------
    psi : array-like
        The quantum state to which the transformation is applied.
    U : array-like
        The unitary matrix representing the transformation.
    decomposition_scheme : str, optional
        The scheme used to decompose the unitary matrix. Options are "rectangular", "rectangular_phase_end", and "triangular".
        Default is "rectangular".
    tol : float, optional
        Tolerance for numerical precision in the decomposition. Default is 1e-11.
    Returns:
    --------
    psi : array-like
        The transformed quantum state.
        """

    if decomposition_scheme.lower() == "rectangular":

        ti_list, diags, t_list = sf.decompositions.rectangular(U, tol)
        apply_interferometer(psi, ti_list, trunc_params = trunc_params, renormalize = renormalize, print_enable = print_enable)
        apply_layer_of_phase_shifter(psi, diags)
        apply_interferometer(psi, reversed(t_list), inverse = True, trunc_params = trunc_params, renormalize = renormalize, print_enable = print_enable)
     
    elif decomposition_scheme.lower() == "rectangular_phase_end":

        t_list, diags, _ = sf.decompositions.rectangular_phase_end(U, tol)
        apply_interferometer(psi, t_list, trunc_params = trunc_params, renormalize = renormalize, print_enable = print_enable)
        apply_layer_of_phase_shifter(psi, diags)
          
    elif decomposition_scheme.lower() == "triangular":

        t_list, diags, _ = sf.decompositions.triangular(U, tol)
        apply_layer_of_phase_shifter(psi, diags)
        apply_interferometer(psi,t_list, inverse = True, trunc_params = trunc_params, renormalize = renormalize, print_enable = print_enable)
    else:
        print("Decomposition scheme not known, using rectangular decomposition")

    return psi

# %%
def generate_SH_model_H(N, J, F_z, Omega_n, M, displacement = False, convention = 'nuesseler'):

    j_i = [4*J] * (N -1)
    A = np.diag(j_i, k = 1) + np.diag(j_i, k = -1)
    A = A.astype(np.complex128)
    B = np.zeros((N, N), dtype = np.complex128)
    
    if displacement:
        #calculate H_R 
        # Step 1: Compute 1 / (2m * Omega_n^2)
        factor = 1 / (2 * Omega_n**2)  # 1D array
        # Step 2: Multiply each column of M_in by factor
        weighted_M = M * factor[np.newaxis, :]  # Broadcasting
        # Step 3: Compute sum_j M_jn (sum over rows of M_in)
        sum_Mjn = np.sum(M, axis=0)  # 1D array of sums over j
        # Step 4: Compute the final prefactor sum over n
        prefactor_array = -4 * F_z**2 * np.sum(weighted_M * sum_Mjn[np.newaxis, :], axis=1)

        A = A + np.diag(prefactor_array)

    # Initialize the Hamiltonian matrix
    H = np.zeros((2*N, 2*N), dtype=np.complex128)
    if convention == 'nuesseler':
        # Fill the blocks of the Hamiltonian matrix obeying the Anticommutation relations
        H[:N, :N] = A
        H[:N, N:] = -B.conj()
        H[N:, :N] = B
        H[N:, N:] = -A.conj()
    elif convention == 'surace':
        # Fill the blocks of the Hamiltonian matrix obeying the Anticommutation relations
        H[:N, :N] = -A.conj()
        H[:N, N:] = B
        H[N:, :N] = -B.conj()
        H[N:, N:] = A
    elif convention == 'serafini':
        # Fill the blocks of the Hamiltonian matrix obeying the Anticommutation relations
        H[:N, :N] = -B.conj()
        H[:N, N:] = -A.conj()
        H[N:, :N] = A
        H[N:, N:] = B
    else:
        raise ValueError("Convention not known. Please choose between 'nuesseler', 'surace' and 'serafini'")
    
    return H
