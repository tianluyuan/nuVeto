import pickle
from functools32 import lru_cache
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import MCEq.core
import MCEq.kernels
import MCEq.density_profiles
import MCEq.data

from MCEq.core import MCEqRun
import CRFluxModels.CRFluxModels as pm
from mceq_config import config, mceq_config_without
from utils import *
from barr_uncertainties import *

class SelfVeto(object):
    def MCEqReconfig(self,config_):
        config_['enable_muon_energy_loss']=False
        config_['debug_level'] = 0
        MCEq.core.dbg = 0
        MCEq.kernels.dbg = 0
        MCEq.density_profiles.dbg = 0
        MCEq.data.dbg = 0
        return config_


    def __init__(self, costh,
                 pmodel=(pm.HillasGaisser2012,'H3a'),
                 hadr='SIBYLL2.3c', barr_mods=(), depth=1950*Units.m):
        """A separate MCEq instance needs to be created for each
        combination of __init__'s arguments. To access pmodel and hadr,
        use mceq.pm_params and mceq.yields_params
        """


        self.costh = costh
        self.pmodel = pmodel
        self.geom = Geometry(depth)
        theta = np.degrees(np.arccos(self.geom.cos_theta_eff(self.costh)))

        self.mceq = MCEqRun(
            # provide the string of the interaction model
            interaction_model=hadr,
            # primary cosmic ray flux model
            # support a tuple (primary model class (not instance!), arguments)
            primary_model=pmodel,
            # zenith angle \theta in degrees, measured positively from vertical direction
            theta_deg = theta,
            **self.MCEqReconfig(config))
        for barr_mod in barr_mods:
            # Modify proton-air -> mod[0]
            self.mceq.set_mod_pprod(2212,BARR[barr_mod[0]].pdg,barr_unc,barr_mod)
        # Populate the modifications to the matrices by re-filling the interaction matrix
        self.mceq._init_default_matrices(skip_D_matrix=True)

        x_vec = np.logspace(np.log10(1e-4),
                            np.log10(self.mceq.density_model.max_X), 11)
        heights = self.mceq.density_model.X2h(x_vec)
        lengths = self.mceq.density_model.geom.delta_l(heights, np.radians(theta)) * Units.cm
        self.dh_vec = np.diff(lengths)
        self.x_vec = centers(x_vec)


    @staticmethod
    def is_prompt(categ):
        return categ == 'pr' or categ[0] in ['D', 'L']

    
    @staticmethod
    def categ_to_mothers(categ, daughter):
        rcharge = '-' if 'anti' in daughter else '+'
        lcharge = '+' if 'anti' in daughter else '-'
        rbar = '-bar' if 'anti' in daughter else ''
        lbar = '' if 'anti' in daughter else '-bar'
        if categ == 'conv':
            mothers = ['pi'+rcharge, 'K'+rcharge, 'K0L']
            if 'nue' in daughter:
                mothers.extend(['K0S', 'mu'+rcharge])
            else:
                mothers.extend(['mu'+lcharge])
        elif categ == 'pr':
            mothers = ['D'+rcharge, 'Ds'+rcharge, 'D0'+rbar]#, 'Lambda0'+lbar]#, 'LambdaC+'+bar]
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


    @lru_cache(2**10)
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


    @lru_cache(maxsize=2**10)
    def grid_sol(self, ecr=None, particle=None):
        if ecr is not None:
            self.mceq.set_single_primary_particle(ecr, particle)
        else:
            self.mceq.set_primary_model(*self.pmodel)
        self.mceq.solve(int_grid=self.x_vec, grid_var="X")
        return self.mceq.grid_sol


    @lru_cache(maxsize=2**10)
    def prob_nomu(self, ecr, particle, prpl='step_1'):
        grid_sol = self.grid_sol(ecr, particle)
        l_ice = self.geom.overburden(self.costh)
        mu = self.get_solution('mu-', grid_sol) + self.get_solution('mu+', grid_sol)

        fn = MuonProb(prpl)
        coords = zip(self.mceq.e_grid*Units.GeV, [l_ice]*len(self.mceq.e_grid))
        return np.exp(-np.trapz(mu*fn.prpl(coords),
                                self.mceq.e_grid))


    def get_solution(self,
                     particle_name,
                     grid_sol,
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
            sol = grid_sol[-1]
            xv = self.x_vec[-1]
        elif grid_idx >= len(self.mceq.grid_sol):
            sol = grid_sol[-1]
            xv = self.x_vec[-1]
        else:
            sol = grid_sol[grid_idx]
            xv = self.x_vec[grid_idx]

        res = np.zeros(len(self.mceq.e_grid))
        part_xs = self.mceq.cs.get_cs(p_pdg)
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
        res[direct!=0] = direct[direct!=0]

        if particle_name[:-1] == 'mu':            
            for _ in ['k_'+particle_name, 'pi_'+particle_name, 'pr_'+particle_name]:
                res += sol[ref[_].lidx():
                           ref[_].uidx()]

        res *= self.mceq.e_grid ** mag

        if not integrate:
            return res
        else:
            return res * self.mceq.e_widths


    def get_rescale_phi(self, mother, grid_sol, idx):
        dh = self.dh_vec[idx]
        inv_decay_length_array = (ParticleProperties.mass_dict[mother] / (self.mceq.e_grid * Units.GeV)) *(dh / ParticleProperties.lifetime_dict[mother])
        rescale_phi = inv_decay_length_array * self.get_solution(mother, grid_sol, grid_idx=idx)
        return interpolate.interp1d(self.mceq.e_grid, rescale_phi, kind='quadratic', fill_value='extrapolate')

    
    def get_integrand(self, categ, daughter, grid_sol, idx, weight_fn, esamp, enu):
        mothers = self.categ_to_mothers(categ, daughter)
        ys = np.zeros(len(esamp))
        for mother in mothers:
            dNdEE = self.get_dNdEE(mother, daughter)[-1]
            rescale_phi = self.get_rescale_phi(mother, grid_sol, idx)
            ys += dNdEE(enu/esamp)/esamp*rescale_phi(esamp)*weight_fn

        return ys


    def get_fluxes(self, enu, kind='conv_numu', accuracy=3, prpl='step_1', corr_only=False):
        categ, daughter = kind.split('_')

        ice_distance = self.geom.overburden(self.costh)

        # TODO: replace 1e8 with MMC-prpl interpolated bounds
        esamp = np.logspace(np.log10(enu),
                            np.log10(enu+1e8), 1000*accuracy)
        identity = np.ones(len(esamp))
        if 'numu' not in daughter:
            # muon accompanies numu only
            reaching = identity
        else:
            fn = MuonProb(prpl)
            reaching = 1. - fn.prpl(zip((esamp-enu)*Units.GeV,
                                        [ice_distance]*len(esamp)))
            if self.is_prompt(categ):
                with np.load(nuVeto.__path__[0]+'/data/decay_distributions/D+_numu.npz') as dfile:
                    xmus = centers(dfile['xedges'])
                    xnus = np.concatenate([xmus, [1]])
                    vals = dfile['histograms']

                    ddec = interpolate.RegularGridInterpolator((xnus, xmus), vals,
                                                               bounds_error=False, fill_value=None)
                    for i, enufrac in enumerate(enu/esamp):
                        emu = xmus*esamp[i]
                        pmu = ddec(zip([enufrac]*len(emu), xmus))
                        reaching[i] = 1 - np.dot(pmu, fn.prpl(zip(emu*Units.GeV,
                                                                  [ice_distance]*len(emu))))

        passed = 0
        total = 0
        if corr_only:
            grid_sol = self.grid_sol()
            for idx in xrange(len(self.x_vec)):
                passed += integrate.trapz(
                    self.get_integrand(categ, daughter, grid_sol, idx, reaching, esamp, enu), esamp)
                total += integrate.trapz(
                    self.get_integrand(categ, daughter, grid_sol, idx, identity, esamp, enu), esamp)
            return passed, total
                
        pmodel = self.pmodel[0](self.pmodel[1])
        for particle in pmodel.nucleus_ids:
            # A continuous input energy range is allowed between
            # :math:`50*A~ \\text{GeV} < E_\\text{nucleus} < 10^{10}*A \\text{GeV}`.
            ecrs = amu(particle)*np.logspace(2, 10, 10*accuracy)
            pnm = [self.prob_nomu(ecr, particle, prpl) for ecr in ecrs]
            pnmfn = interpolate.interp1d(ecrs, pnm, kind='cubic',
                                         assume_sorted=True, bounds_error=False,
                                         fill_value=(1,np.nan))
            nums = []
            dens = []
            istart = max(0, np.argmax(ecrs > enu) - 1)
            for ecr in ecrs[istart:]:
                cr_flux = pmodel.nucleus_flux(particle, ecr)*Units.phim2
                # poisson exp(-Nmu)
                pnmarr = pnmfn(ecr-esamp)
                # cubic splining doesn't enforce 0-1 bounds
                pnmarr[pnmarr>1] = 1
                pnmarr[pnmarr<0] = 0
                # print pnmarr
                grid_sol = self.grid_sol(ecr, particle)
                num_ecr = 0
                den_ecr = 0
                # dh
                for idx in xrange(len(self.x_vec)):
                    # dEp
                    num_ecr += integrate.trapz(
                        self.get_integrand(
                            categ, daughter, grid_sol, idx,
                            reaching, esamp, enu)*pnmarr, esamp)
                    den_ecr += integrate.trapz(
                        self.get_integrand(
                            categ, daughter, grid_sol, idx,
                            identity, esamp, enu), esamp)

                nums.append(num_ecr*cr_flux/Units.phicm2)
                dens.append(den_ecr*cr_flux/Units.phicm2)
            # dEcr
            passed += integrate.trapz(nums, ecrs[istart:])
            total += integrate.trapz(dens, ecrs[istart:])

        return passed, total


@lru_cache(maxsize=2**10)
def builder(cos_theta, pmodel, hadr, barr_mods, depth):
    return SelfVeto(cos_theta, pmodel, hadr, barr_mods, depth)


def passing(enu, cos_theta, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', barr_mods=(), depth=1950*Units.m, accuracy=3, fraction=True, prpl='step_1', corr_only=False):
    sv = builder(cos_theta, pmodel, hadr, barr_mods, depth)
    num, den = sv.get_fluxes(enu, kind, accuracy, prpl, corr_only)
    return num/den if fraction else num


def total(enu, cos_theta, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', barr_mods=(), depth=1950*Units.m, accuracy=3, corr_only=False):
    sv = builder(cos_theta, pmodel, hadr, barr_mods, depth)
    return sv.get_fluxes(enu, kind, accuracy, corr_only=corr_only)[1]
