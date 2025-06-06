import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.networks import site
import numpy as np


#computation time is O(n^2 + n) this way.
def compute_elasticity_matrix(N, beta, c, w_z = 1):
    
    """
    Takes the parameters N, beta and c as input
    
    N: number of sites
    beta: stiffness parameter
    c : proportional factor
    
    returns:
    K : Elasticity matrix 
    """
    K = np.zeros((N, N))  # Initialize an N x N matrix with zeros

    # Precompute diagonal sums as then we dont have to run the sum over k all over as this will not change 
    diagonal_sum = np.zeros(N)
    for i in range(N):
        Sum = sum(beta / (abs(i - k)**3) for k in range(N) if k != i)
        diagonal_sum[i] = Sum #vector with the term to add to K[i][i]

    # Fill matrix K
    for i in range(N):
        K[i][i] = 1 + c * diagonal_sum[i]  # Diagonal elements
        for j in range(i + 1, N):
            # Compute only for i < j
            value = -c * beta / (abs(i - j)**3)
            K[i][j] = value  # Off-diagonal element
            K[j][i] = value  # Exploit symmetry, as K[i][j] = K[j][i]

    return  (w_z**2)*K


def compute_omegas_and_g(K, displacement = False):
    """
    Computes the normal mode frequencies (omegas) and coupling constants (g) from a force constant matrix K.
    This function performs eigendecomposition of the force constant matrix to obtain
    the normal modes and their frequencies. It then calculates the coupling constants
    according to the transformation g = M.T / sqrt(2*omega), where M is the matrix of
    eigenvectors.
    Parameters
    ----------
    K : numpy.ndarray
        Force constant matrix (should be symmetric and positive definite)
    Returns
    -------
    omegas : numpy.ndarray
        Normal mode frequencies (square root of eigenvalues)
    g : numpy.ndarray
        Transformed coupling constants matrix
    m : numpy.ndarray
        Matrix of eigenvectors
    Raises
    ------
    AssertionError
        If the diagonalization verification fails (M.T @ K @ M ≠ diag(eigenvalues))
    Notes
    -----
    The function verifies the diagonalization by checking if M.T @ K @ M equals
    the diagonal matrix of eigenvalues (within rounding to 1 decimal place).
    """
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        g = np.empty(K.shape)
        m = eigenvectors

        omegas = np.sqrt(eigenvalues) #omegas are the sqaure root of the eigenvalues

        # Precompute the denominator
        denominator = np.sqrt(2 * omegas)  # Shape: (N,)
        #denominator = np.sqrt(omegas)  # Shape: (N,) #for comparison 

        # Divide each row of eigenvectors.T by the corresponding value in the denominator
        g = m.T / (denominator[:, np.newaxis])  # Broadcasting the denominator 

        #test if matrix diagonalization worked via equation S6
        identity = np.round((m.T @ K @ m)/(eigenvalues), 1)
        assert  np.all(identity == np.identity(K.shape[0])) == True, "Diagonalization failed. Check input again!"
        
        if displacement:
            return omegas, g.T, m
        else:
            return omegas, g.T

    except AssertionError as msg:
        print(msg) 

class Spin_Holstein_Model(CouplingMPOModel):
    """Spin Holstein Model for a 1D chain. Periodic boundary conditions are at the moment not implemented"""

    default_lattice = "Chain"
    force_default_lattice = True

    def init_sites(self, model_params):
        spin = site.SpinHalfSite(conserve='Sz')
        boson = site.BosonSite(Nmax = model_params.get('Nmax', 5), conserve=None)
        sites = [spin, boson]
        site.set_common_charges(sites, new_charges='independent')
        return [spin, boson], ['s', 'b']

    def init_terms(self, model_params):
        #extract parameters
        L = model_params.get('L', 10)  # Number of sites
        J = model_params.get('J', 1.)  # Spin coupling
        F_z = model_params.get('F_z', 1.)  # Spin-boson coupling strength
        beta = model_params.get('beta', 1.) #stiffness of the trap
        c_z = model_params.get('c_z', 2)
        energy_shift = model_params.get('energy_shift', True)
        displacement = model_params.get('displacement', True) #change to false when combining to one big class
        if displacement == True:
            Omega_n, g_in, M_in = compute_omegas_and_g(compute_elasticity_matrix(L, beta, c_z), displacement) # Collective bosonic mode frequencies
        else:
            Omega_n, g_in = compute_omegas_and_g(compute_elasticity_matrix(L, beta, c_z), displacement) # Collective bosonic mode frequencies
            
        L_cutoff = model_params.get('L_cutoff', L)
        
        
        # Phonon modes
        self.add_onsite(Omega_n, 1, 'N')
        if energy_shift:
            self.add_onsite_term(0.5*np.sum(Omega_n), 1, 'Id')
        
        
        # spin-spin interaction in x-y
        for l in range(1, L_cutoff): 
            coupling_strength = 2*J/l**3 #factor 2 through the convention that S_p = 1/2 (simga_x + i sigma_y)
            coupling_strength = coupling_strength/4
            self.add_coupling(coupling_strength, 0, 'Sp', 0, 'Sm', l, plus_hc = True) #term for i > j
            self.add_coupling(coupling_strength, 0, 'Sp', 0, 'Sm', -l, plus_hc = True) #term for i < j
            
        
        #Implementation of H_int up to the end            
        
        # Assuming the matrix is square (nxn), get shape of dimension
        n = g_in.shape[0] 
        
        # List to store all diagonals
        strengths = []
        # Get all diagonals from above the main diagonal (k > 0), main diagonal (k=0), and below (k < 0)
        for k in range(-n + 1, n):
            strengths.append(-F_z * np.diag(g_in, k=k))
        
        #set distance 0 terms 
        self.add_coupling(strengths[n-1], 1, 'B', 0, 'Sigmaz', 0)
        self.add_coupling(strengths[n-1], 1, 'Bd', 0, 'Sigmaz', 0)
        
    
        for dist in range(1, L_cutoff):
            #adding a_n sigma_n^z terms 
            self.add_coupling(strengths[n - 1 - dist], 1, 'B', 0, 'Sigmaz', dist)
            self.add_coupling(strengths[n - 1 + dist], 1, 'B', 0, 'Sigmaz', -dist)
            
            self.add_coupling(strengths[n - 1 - dist], 1, 'Bd', 0, 'Sigmaz', dist)
            self.add_coupling(strengths[n - 1  + dist], 1, 'Bd', 0, 'Sigmaz', -dist)
            
        if displacement == True:
             #Hr = residual term -- only change NOTE:  YOu have to add the constant term afterwards to your found GS energy!
            residual_coupling = np.zeros(L)
        
            for i in range(L_cutoff):
                summation  = 0
                for n in range(L):
                    sum_M_jn = np.sum(M_in[:, n]) #sm over j for each n
                    M_in_over_omega_n = M_in[i, n] / (2 * Omega_n[n] ** 2)
                    summation += M_in_over_omega_n * sum_M_jn
             
           
                residual_coupling[i] = - 2 * F_z ** 2 * summation

            self.add_onsite(residual_coupling, 0, 'Sigmaz')
            
            #add extra constant term
            HR_1 = 0
            # Iterate over each n (second axis of M_in)
            for n in range(L_cutoff):
                # Sum over i for M_in[:,n] (all elements in the nth column)
                sum_M_in = np.sum(M_in[:, n])

                # Add the contribution of this n to H_R
                HR_1 += (1 / (2 * Omega_n[n]**2)) * (sum_M_in ** 2)

            # Apply the constant factor F_z^2
            HR_1 = -F_z**2 * HR_1
            
            self.add_onsite_term(HR_1, 0, "Id")
        else:
             for i in range(L_cutoff):
                coupling_phonon_spins = -F_z * g_in[i]
                self.add_onsite(coupling_phonon_spins, 1, 'B')
                self.add_onsite(coupling_phonon_spins, 1, 'Bd')
            