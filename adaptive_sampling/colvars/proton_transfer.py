import torch

from .utils import *
from ..units import * 

class PT:
    """ Proton Transfer (PT) Collective Variables

    see: König et. al, J. Phys. Chem. A, (2006): https://doi.org/10.1021/jp052328q
    
    Args:
        r_sw: switching distance in Angstrom, Atoms closer than r_sw are considered coordinated
        d_sw: in Angstrom, controls how fast switching function flips from 0 to 1
        n_pair: exponent for calculation of m(X,{H})
        requires_grad: if gradient of CV should be calculated
    """

    def __init__(
        self, 
        r_sw: float=1.4, 
        d_sw: float=0.05, 
        n_pair: int=15,
        requires_grad: bool=True,      
    ):
        self.r_sw          = r_sw / BOHR_to_ANGSTROM
        self.d_sw          = d_sw / BOHR_to_ANGSTROM
        self.n_pair        = n_pair
        self.requires_grad = requires_grad
        self.cv            = None
        self.gradient      = None

    def cec(
        self, 
        coords: torch.tensor, 
        proton_idx: list, 
        heavy_idx: list, 
        heavy_weights: list, 
        ref_idx: list, 
        pair_def: list=[], 
        modified: bool = True,
    ) -> torch.tensor:
        """Center of Excess Charge coordinate for long-range PT 
        projected on 1D axis of PT

        Args:
            coords: cartesian coordinates
            proton_idx: list od indices of protons
            heavy_idx: list of indices of heavy atoms
            heavy_weights: list of weights of heavy atoms
            ref_idx: Atom indices that define axis of PT (usually donor and acceptor atom)
            pair_def: list of lists of indices of coupled atom pairs: [[weight_pair, idx0, idx1], ...]
            modified: if True, use CEC modification by König et al.

        Returns:
            cv: CEC coordinate
        """
        z = coords.view(int(torch.numel(coords) / 3), 3)
        self.cv = 0.0

        # vector that defines 1D direction of proton transport
        r_don = z[ref_idx[0]]
        r_acc = z[ref_idx[1]]

        z_pt = r_acc - r_don
        z_n  = 1. / torch.linalg.norm(z_pt)
        z_u  = z_pt * z_n

        # sum over protons
        for _, idx_h in enumerate(proton_idx):
            self.cv += torch.matmul(z[idx_h] - r_don, z_u)

        # sum over donors/acceptors
        for _, (idx_x, w_x) in enumerate(zip(heavy_idx, heavy_weights)): 
            self.cv -= w_x * torch.matmul(z[idx_x] - r_don, z_u)
            
        # mixed sum for modified CEC
        if modified:
            for _, idx_hi in enumerate(proton_idx):
                r_hi = z[idx_hi]
                for _, idx_xj in enumerate(heavy_idx):
                    r_ij = r_hi - z[idx_xj]
                    self.cv -= self._f_sw(r_ij) * torch.matmul(r_ij, z_u)

        # correction for coupled donor and acceptor 
        # (e.g. for glutamate, aspartate, histidine, ...)
        if bool(pair_def):
            w_pair = [j[0] for j in pair_def]
            ind_pair = [[j[1], j[2]] for j in pair_def]
            for _, (w_pj, ind_pj) in enumerate(zip(w_pair, ind_pair)):
                
                r_k = z[ind_pj[0]]
                r_l = z[ind_pj[1]]
                
                r_kl = torch.matmul(r_l - r_k, z_u)

                # accumulators for m_k and m_l
                denom_k, nom_k = 0.0, 0.0
                denom_l, nom_l = 0.0, 0.0

                # compute m_k and m_l as sum over protons
                for _, idx_hi in enumerate(proton_idx):
                    r_hi = z[idx_hi]
                    f_k = self._f_sw(r_hi - r_k)
                    f_l = self._f_sw(r_hi - r_l)

                    # for heavy atoms sum over all protons contributes to gradient
                    nom_k   += torch.pow(f_k, self.n_pair + 1)
                    nom_l   += torch.pow(f_l, self.n_pair + 1)
                    denom_k += torch.pow(f_k, self.n_pair)
                    denom_l += torch.pow(f_l, self.n_pair)

                # add coupled term to xi
                m_k = nom_k / denom_k
                m_l = nom_l / denom_l
                self.cv += (w_pj / 2.0) * (m_k * r_kl - m_l * r_kl)

        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, coords, allow_unused=True
            )[0]
            self.gradient.detach().numpy()

        return self.cv

    def gmcec(
        self, 
        coords: torch.tensor, 
        proton_idx: list, 
        heavy_idx: list, 
        heavy_weights: list,
        ref_idx: list, 
        pair_def: list=[], 
        mapping: str = "default"
    ) -> torch.tensor:
        """Generalized modified CEC coordinate to describe long range proton transfer generalized to complex 3D wire geometries after König et al.

        Args:
            coords: cartesian coordinates
            proton_idx: list od indices of protons
            heavy_idx: list of indices of heavy atoms
            heavy_weights: list of weights of heavy atoms
            ref_idx: reference indices for mapping of CV (usually donor and acceptor atom)
            pair_def: list of lists of coupled atom pairs: [[weight_pair, idx0, idx1], ...]
            mapping: 'f_SW': mapping to switching function (chi = 1/(1+(1+e^(d_acc_xi))) - 1/(1+e^(d_don_xi)))
                     'fraction': chi = d_don_xi / (d_don_xi+d_acc_xi)
                     'default': antisymmetric stretch between ref_idx[0], xi and ref_idx[1] (chi = (d_xi_ref0 - d_xi_ref1) / 2.)
                         
        Returns:
            cv: gmCEC coordinate
        """
        z  = coords.view(int(torch.numel(coords) / 3), 3)
        xi = torch.zeros(3, dtype=torch.float)

        # protons
        for _, idx_h in enumerate(proton_idx):
            xi += z[idx_h]

        # donors/acceptors
        for _, (idx_x, w_x) in enumerate(zip(heavy_idx, heavy_weights)):
            xi -= w_x * z[idx_x]

        # modified CEC
        for _, idx_h in enumerate(proton_idx):
            r_hi = z[idx_h]
            for _, idx_x in enumerate(heavy_idx):
                r_ij = (r_hi - z[idx_x])
                xi -= self._f_sw(r_ij) * r_ij

        # correction for coupled donor and acceptor 
        # (e.g. for glutamate, aspartate, histidine, ...)
        if bool(pair_def):
            w_pair = [j[0] for j in pair_def]
            index_pair = [[j[1], j[2]] for j in pair_def]
            for _, (w_pj, idx_pj) in enumerate(zip(w_pair, index_pair)):
                
                r_k = z[idx_pj[0]]
                r_l = z[idx_pj[1]]
                r_kl = r_l - r_k

                # accumulators for m_k and m_l
                denom_k, num_k = 0.0, 0.0
                denom_l, num_l = 0.0, 0.0

                # compute m_k, m_l and their derivatives
                for _, r_hi in enumerate(proton_idx):
                    r_ki = r_hi - r_k
                    r_li = r_hi - r_l
                    f_k = self._f_sw(r_ki)
                    f_l = self._f_sw(r_li)

                    # for heavy atoms sum over all protons contributes to gradient
                    denom_k += torch.pow(f_k, self.n_pair)
                    denom_l += torch.pow(f_l, self.n_pair)
                    num_k   += torch.pow(f_k, self.n_pair + 1)
                    num_l   += torch.pow(f_l, self.n_pair + 1)

                # add coupled term to xi
                m_k = num_k / denom_k
                m_l = num_l / denom_l
                xi +=  (w_pj / 2.0) * (m_k * r_kl - m_l * r_kl)

        # mapping to 1D
        if mapping.lower() == "f_sw":
            self.cv =- 1.0 / (1. + torch.exp(torch.linalg.norm(xi - z[ref_idx[0]])))
            self.cv += 1.0 / (1. + torch.exp(torch.linalg.norm(xi - z[ref_idx[1]])))

        elif mapping.lower() == "fraction":
            d_xi_don = torch.linalg.norm(xi - z[ref_idx[0]])
            d_xi_acc = torch.linalg.norm(xi - z[ref_idx[1]])
            self.cv = d_xi_don / (d_xi_don + d_xi_acc)

        else: # default
            self.cv = 0.5 * (torch.linalg.norm(xi - z[ref_idx[0]]) - torch.linalg.norm(xi - z[ref_idx[1]]))
        
        # gradient of gmcec coordinate
        if self.requires_grad:
            self.gradient = torch.autograd.grad(
                self.cv, coords, allow_unused=True
            )[0]
            self.gradient.detach().numpy()

        return self.cv

    def _f_sw(self, r) -> float:
        """switching function f_sw(r)"""
        return 1. / (1. + torch.exp((torch.linalg.norm(r) - self.r_sw) / self.d_sw))
