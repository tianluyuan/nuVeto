from functools32 import lru_cache
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from MCEq.core import MCEqRun
import CRFluxModels as pm
from mceq_config import config, mceq_config_without
from utils import *


class SelfVeto(object):
    def __init__(self, costh,
                 pmodel=(pm.HillasGaisser2012,'H3a'),
                 hadr='SIBYLL2.3c'):
        """A separate MCEq instance needs to be created for each
        combination of __init__'s arguments. To access pmodel and hadr,
        use mceq.pm_params and mceq.yields_params
        """
        self.costh = costh
        self.geom = Geometry(1950*Units.m)
        theta = np.degrees(np.arccos(self.geom.cos_theta_eff(self.costh)))

        self.mceq = MCEqRun(
            # provide the string of the interaction model
            interaction_model=hadr,
            # primary cosmic ray flux model
            # support a tuple (primary model class (not instance!), arguments)
            primary_model=pmodel,
            # zenith angle \theta in degrees, measured positively from vertical direction
            theta_deg = theta,
            # expand the rest of the options from mceq_config.py
            **config)

        x_vec = np.logspace(np.log10(1e-4),
                            np.log10(self.mceq.density_model.max_X), 11)
        heights = self.mceq.density_model.X2h(x_vec)
        lengths = self.mceq.density_model.geom.delta_l(heights, np.radians(theta)) * Units.cm
        self.dh_vec = np.diff(lengths)
        self.x_vec = x_vec[:-1]
        self.mceq.solve(int_grid=self.x_vec, grid_var="X")


    @staticmethod
    def categ_to_mothers(categ, daughter):
        charge = '-' if 'anti' in daughter else '+'
        lcharge = '+' if 'anti' in daughter else '-'
        bar = '-bar' if 'anti' in daughter else ''
        lbar = '' if 'anti' in daughter else '-bar'
        if categ == 'conv':
            mothers = ['pi'+charge, 'K'+charge, 'K0L']
            if 'nue' in daughter:
                mothers.extend(['K0S', 'mu'+charge])
            else:
                mothers.extend(['mu'+lcharge])
        elif categ == 'pr':
            mothers = ['D'+charge, 'Ds'+charge, 'D0'+bar, 'Lambda0'+lbar]# 'LambdaC+'+bar
        else:
            mothers = [categ,]
        return mothers


    @staticmethod
    def projectiles():
        pdg_ids = config['adv_set']['allowed_projectiles']
        namer = ParticleProperties.modtab.pdg2modname
        allowed = []
        for pdg_id in pdg_ids:
            allowed.append(namer[pdg_id])
            try:
                allowed.append(namer[-pdg_id])
            except KeyError:
                continue
        return allowed


    def get_dNdEE(self, mother, daughter):
        ihijo = 20
        e_grid = self.mceq.e_grid
        delta = self.mceq.e_widths
        x_range = e_grid[ihijo]/e_grid
        rr = ParticleProperties.rr(mother, daughter)
        dNdEE_edge = ParticleProperties.br_2body(mother, daughter)/(1-rr)
        dN_mat = self.mceq.decays.get_d_matrix(
            ParticleProperties.pdg_id[mother],
            ParticleProperties.pdg_id[daughter])
        dNdEE = dN_mat[ihijo]*e_grid/delta
        logx = np.log10(x_range)
        logx_width = -np.diff(logx)[0]
        good = (logx + logx_width/2 < np.log10(1-rr)) & (x_range >= 5.e-2)

        lower = dNdEE[good][-1]
        dNdEE_interp = interpolate.interp1d(
            np.concatenate([[1-rr], x_range[good]]),
            np.concatenate([[dNdEE_edge], dNdEE[good]]), kind='quadratic',
            bounds_error=False, fill_value=(lower, 0.0))
        return x_range, dNdEE, dNdEE_interp


    def get_solution_orig(self,
                          particle_name,
                          mag=0.,
                          grid_idx=None,
                          integrate=False):
        """Retrieves solution of the calculation on the energy grid.

        Some special prefixes are accepted for lepton names:

        - the total flux of muons, muon neutrinos etc. from all sources/mothers
          can be retrieved by the prefix ``total_``, i.e. ``total_numu``
        - the conventional flux of muons, muon neutrinos etc. from all sources
          can be retrieved by the prefix ``conv_``, i.e. ``conv_numu``
        - correspondigly, the flux of leptons which originated from the decay
          of a charged pion carries the prefix ``pi_`` and from a kaon ``k_``
        - conventional leptons originating neither from pion nor from kaon
          decay are collected in a category without any prefix, e.g. ``numu`` or
          ``mu+``

        Args:
          particle_name (str): The name of the particle such, e.g.
            ``total_mu+`` for the total flux spectrum of positive muons or
            ``pr_antinumu`` for the flux spectrum of prompt anti muon neutrinos
          mag (float, optional): 'magnification factor': the solution is
            multiplied by ``sol`` :math:`= \\Phi \\cdot E^{mag}`
          grid_idx (int, optional): if the integrator has been configured to save
            intermediate solutions on a depth grid, then ``grid_idx`` specifies
            the index of the depth grid for which the solution is retrieved. If
            not specified the flux at the surface is returned
          integrate (bool, optional): return averge particle number instead of
          flux (multiply by bin width)

        Returns:
          (numpy.array): flux of particles on energy grid :attr:`e_grid`
        """
        res = np.zeros(self.mceq.d)
        ref = self.mceq.pname2pref
        sol = None
        if grid_idx is None:
            sol = self.mceq.grid_sol[-1]
        elif grid_idx >= len(self.mceq.grid_sol):
            sol = self.mceq.grid_sol[-1]
        else:
            sol = self.mceq.grid_sol[grid_idx]

        if particle_name.startswith('total'):
            lep_str = particle_name.split('_')[1]
            for prefix in ('pr_', 'pi_', 'k_', ''):
                particle_name = prefix + lep_str
                res += sol[ref[particle_name].lidx():
                           ref[particle_name].uidx()] * \
                    self.mceq.e_grid ** mag
        elif particle_name.startswith('conv'):
            lep_str = particle_name.split('_')[1]
            for prefix in ('pi_', 'k_', ''):
                particle_name = prefix + lep_str
                res += sol[ref[particle_name].lidx():
                           ref[particle_name].uidx()] * \
                    self.mceq.e_grid ** mag
        else:
            res = sol[ref[particle_name].lidx():
                      ref[particle_name].uidx()] * \
                self.mceq.e_grid ** mag

        if not integrate:
            return res
        else:
            return res * self.mceq.e_widths


    def get_solution(self,
                     particle_name,
                     mag=0.,
                     grid_idx=None,
                     integrate=False):
        """Retrieves solution of the calculation on the energy grid.

        Args:
          particle_name (str): The name of the particle such, e.g.
            ``total_mu+`` for the total flux spectrum of positive muons or
            ``pr_antinumu`` for the flux spectrum of prompt anti muon neutrinos
          mag (float, optional): 'magnification factor': the solution is
            multiplied by ``sol`` :math:`= \\Phi \\cdot E^{mag}`
          grid_idx (int, optional): if the integrator has been configured to save
            intermediate solutions on a depth grid, then ``grid_idx`` specifies
            the index of the depth grid for which the solution is retrieved. If
            not specified the flux at the surface is returned
          integrate (bool, optional): return averge particle number instead of
          flux (multiply by bin width)

        Returns:
          (numpy.array): flux of particles on energy grid :attr:`e_grid`
        """
        res = np.zeros(self.mceq.d)
        ref = self.mceq.pname2pref
        sol = None
        p_pdg = ParticleProperties.pdg_id[particle_name]
        if grid_idx is None:
            sol = self.mceq.grid_sol[-1]
        elif grid_idx >= len(self.mceq.grid_sol):
            sol = self.mceq.grid_sol[-1]
        else:
            sol = self.mceq.grid_sol[grid_idx]

        res = np.zeros(len(self.mceq.e_grid))
        part_xs = self.mceq.cs.get_cs(p_pdg)
        xv = self.x_vec[grid_idx]
        rho_air = self.mceq.density_model.X2rho(xv)
        # meson decay length
        decayl = (self.mceq.e_grid * Units.GeV)/ParticleProperties.mass_dict[particle_name] * ParticleProperties.lifetime_dict[particle_name] /Units.cm
        # meson interaction length
        interactionl = 1/(self.mceq.cs.get_cs(p_pdg)*rho_air*Units.Na/Units.mol_air)
        # number of targets per cm2
        ndens = rho_air*Units.Na/Units.mol_air
        for prim in self.projectiles():
            prim_flux = sol[ref[prim].lidx():
                            ref[prim].uidx()]
            prim_xs = self.mceq.cs.get_cs(ParticleProperties.pdg_id[prim])
            try:
                int_yields = self.mceq.y.get_y_matrix(
                    ParticleProperties.pdg_id[prim],
                    p_pdg)
                res += np.dot(int_yields,
                              prim_flux*prim_xs*ndens)
            except KeyError as e:
                continue
                
        res *= decayl
        # combine with direct
        direct = sol[ref[particle_name].lidx():
                     ref[particle_name].uidx()]
        res = np.concatenate((res[direct==0], direct[direct!=0]))

        if particle_name[:-1] == 'mu':            
            for _ in ['k_'+particle_name, 'pi_'+particle_name, 'pr_'+particle_name]:
                res += sol[ref[_].lidx():
                           ref[_].uidx()]
        elif particle_name.startswith('K'):
            for _ in self.mceq.decays.mothers:
                if _%1e3 == 7:
                    continue
                if self.mceq.decays.is_daughter(_,p_pdg):
                    namer = ParticleProperties.modtab.pdg2modname
                    direct = sol[ref[namer[_]].lidx():
                                 ref[namer[_]].uidx()]
                    res+=np.dot(self.mceq.decays.get_d_matrix(_,
                                                              p_pdg),
                                (self.get_solution(namer[_], grid_idx=grid_idx)-direct)*self.mceq.e_widths)/self.mceq.e_widths

        res *= self.mceq.e_grid ** mag

        if not integrate:
            return res
        else:
            return res * self.mceq.e_widths


    def get_rescale_phi(self, mother, idx):
        dh = self.dh_vec[idx]
        inv_decay_length_array = (ParticleProperties.mass_dict[mother] / (self.mceq.e_grid * Units.GeV)) *(dh / ParticleProperties.lifetime_dict[mother])
        rescale_phi = inv_decay_length_array * self.get_solution(mother, grid_idx=idx)
        return interpolate.interp1d(self.mceq.e_grid, rescale_phi, kind='quadratic', fill_value='extrapolate')

    
    def get_integrand(self, categ, daughter, idx, weight_fn, esamp, enu):
        mothers = self.categ_to_mothers(categ, daughter)
        ys = np.zeros(len(esamp))
        for mother in mothers:
            dNdEE = self.get_dNdEE(mother, daughter)[-1]
            rescale_phi = self.get_rescale_phi(mother, idx)
            ys += dNdEE(enu/esamp)/esamp*rescale_phi(esamp)*weight_fn(esamp)

        return ys


    def get_fluxes(self, enu, kind='conv_numu', accuracy=4, prpl='step_1'):
        categ, daughter = kind.split('_')

        ice_distance = self.geom.overburden(self.costh)
        identity = lambda Ep: 1
        if 'numu' not in daughter:
            # muon accompanies numu only
            reaching = identity
        else:
            fn = MuonProb(prpl)
            reaching = lambda Ep: 1. - fn.prpl(zip((Ep-enu)*Units.GeV,
                                                   [ice_distance]*len(Ep)))

        passing_numerator = 0
        passing_denominator = 0
        esamp = np.logspace(np.log10(enu), np.log10(self.mceq.e_grid[-1]), int(10**accuracy))
        for idx in xrange(len(self.x_vec)):
            passing_numerator += integrate.trapz(
                self.get_integrand(categ, daughter, idx, reaching, esamp, enu), esamp)
            passing_denominator += integrate.trapz(
                self.get_integrand(categ, daughter, idx, identity, esamp, enu), esamp)
            # print passing_numerator, passing_denominator
        return passing_numerator, passing_denominator


SVS = {}


def passing_rate(enu, cos_theta, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', accuracy=4, fraction=True, prpl='step_1'):
    try:
        sv = SVS[(cos_theta, pmodel, hadr)]
    except KeyError:        
        sv = SelfVeto(cos_theta, pmodel, hadr)
        SVS[(cos_theta, pmodel, hadr)] = sv

    num, den = sv.get_fluxes(enu, kind, accuracy, prpl)
    return num/den if fraction else num


def total_flux(enu, cos_theta, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', accuracy=4, prpl='step_1'):
    try:
        sv = SVS[(cos_theta, pmodel, hadr)]
    except KeyError:        
        sv = SelfVeto(cos_theta, pmodel, hadr)
        SVS[(cos_theta, pmodel, hadr)] = sv

    return sv.get_fluxes(enu, kind, accuracy, prpl)[1]
