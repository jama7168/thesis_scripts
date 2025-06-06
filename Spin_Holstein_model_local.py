# %%
import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.networks import site
import numpy as np

# %%
class Spin_Holstein_Model_local(CouplingMPOModel):
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
        #L_cutoff = model_params.get('L_cutoff', L)
        w_z = model_params.get('w_z', 1)
        energy_shift = model_params.get('energy_shift', True)


        #calculate frequencies and couplings
        # w_i from equation (5) from holtein_ions file
        w_i = np.zeros(L)
        for i in range(L):
            Sum = sum(beta / (abs(i - k)**3) for k in range(L) if k != i)
            w_i[i] = w_z* np.sqrt(1 + c_z*Sum) 
        
        # t_ij from equation (6) from holtein_ions file
        t = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                if i != j:
                    denominator = 2*abs(i - j)**3 * np.sqrt(w_i[i] * w_i[j])
                    t[i][j] = -(w_z**2) * beta / denominator
        
        
        # Phonon modes
        self.add_onsite(w_i, 1, 'N')
        if energy_shift:
            self.add_onsite_term(0.5*np.sum(w_i), 1, 'Id')
       
        # List to store all diagonals
        couplings = []
        n = t.shape[0] 
        # Get all diagonals from above the main diagonal (k > 0), main diagonal (k=0), and below (k < 0)
        for k in range(-n + 1, n):
            couplings.append(np.diag(t, k=k))

        for dist in range(1, L):
            self.add_coupling(couplings[n-1+ dist], 1, 'B', 1, 'B', dist, plus_hc = True) #term for i < j
            self.add_coupling(couplings[n-1 - dist], 1, 'B', 1, 'B', -dist, plus_hc = True) #term for i > j

            self.add_coupling(couplings[n-1 +dist], 1, 'B', 1, 'Bd', dist, plus_hc = True) #term for i < j
            self.add_coupling(couplings[n-1 -dist], 1, 'B', 1, 'Bd', -dist, plus_hc = True) #term for i > j
        
    
        # spin-spin interaction in x-y
        for dist in range(1, 2): 
            coupling_strength = 2*J/dist**3 # check for factor 2 (possible wrong here)
            coupling_strength = coupling_strength/4
            self.add_coupling(coupling_strength, 0, 'Sp', 0, 'Sm', dist, plus_hc = True) #term for i < j
            self.add_coupling(coupling_strength, 0, 'Sp', 0, 'Sm', -dist, plus_hc = True) #term for i > j
            
        
        #Implementation of H_int up to the end            

        #calculate g
        g = -1*F_z/np.sqrt(2* w_i)
        #add interaction terms 
        self.add_onsite(g, 1, 'B', plus_hc = True) # g(b ) + g( b^dagger)
        self.add_coupling(g, 1, 'B', 0, 'Sigmaz', 0) #g(b Sigmaz)
        self.add_coupling(g, 1, 'Bd', 0, 'Sigmaz', 0) #g(b_dagger Sigmaz)
