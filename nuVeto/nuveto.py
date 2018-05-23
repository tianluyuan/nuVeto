"""nuVeto calculation [ref.]

This module computes the probability that an atmospheric neutrino will be
accompanied by a sibling muon produced in the same cosmic ray airshower at a
given depth.


"""

from functools32 import lru_cache
from pkg_resources import resource_filename
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import MCEq.core
import MCEq.kernels
import MCEq.density_profiles
import MCEq.data

from MCEq.core import MCEqRun
try:
    import CRFluxModels.CRFluxModels as pm
except ImportError:
    import CRFluxModels as pm
from mceq_config import config, mceq_config_without
from nuVeto.utils import Units, ParticleProperties, MuonProb, Geometry, amu, centers
from nuVeto.barr_uncertainties import BARR, barr_unc

class nuVeto(object):
    """Class for computing the neutrino passing fraction i.e. (1-(Veto probability))"""
    def __init__(self, costh,
                 pmodel=(pm.HillasGaisser2012, 'H3a'),
                 hadr='SIBYLL2.3c', barr_mods=(), depth=1950*Units.m,
                 density=('CORSIKA', ('SouthPole', 'June'))):
        """Initializes the nuVeto object for a particular costheta, CR Flux,
        hadronic model, barr parameters, and depth

        Note:
            A separate MCEq instance needs to be created for each
            combination of __init__'s arguments. To access pmodel and hadr,
            use mceq.pm_params and mceq.yields_params
        Args:
            costh (float): Cos(theta), the cosine of the neutrino zenith at the detector
            pmodel (tuple(CR model class, arguments)): CR Flux
            hadr (str): hadronic interaction model
            barr_mods: barr parameters
            depth (float): the depth at which the veto probability is computed below the ice
        """
        self.costh = costh
        self.pmodel = pmodel
        self.geom = Geometry(depth)
        theta = np.degrees(np.arccos(self.geom.cos_theta_eff(self.costh)))

        MCEq.core.dbg = 0
        MCEq.kernels.dbg = 0
        MCEq.density_profiles.dbg = 0
        MCEq.data.dbg = 0
        self.mceq = MCEqRun(
            # provide the string of the interaction model
            interaction_model=hadr,
            # atmospheric density model
            density_model=density,
            # primary cosmic ray flux model
            # support a tuple (primary model class (not instance!), arguments)
            primary_model=pmodel,
            # zenith angle \theta in degrees, measured positively from vertical direction
            theta_deg=theta,
            enable_muon_energy_loss=False,
            **mceq_config_without(['enable_muon_energy_loss', 'density_model']))

        for barr_mod in barr_mods:
            # Modify proton-air -> mod[0]
            self.mceq.set_mod_pprod(2212, BARR[barr_mod[0]].pdg, barr_unc, barr_mod)
        # Populate the modifications to the matrices by re-filling the interaction matrix
        self.mceq._init_default_matrices(skip_D_matrix=True)

        X_vec = np.logspace(np.log10(2e-3),
                            np.log10(self.mceq.density_model.max_X), 11)
        self.dX_vec = np.diff(X_vec)
        self.X_vec = 10**centers(np.log10(X_vec))


    @staticmethod
    def categ_to_mothers(categ, daughter):
        """Get the parents for this category"""
        rcharge = '-' if 'anti' in daughter else '+'
        lcharge = '+' if 'anti' in daughter else '-'
        rbar = '-bar' if 'anti' in daughter else ''
        #lbar = '' if 'anti' in daughter else '-bar'
        if categ == 'conv':
            mothers = ['pi'+rcharge, 'K'+rcharge, 'K0L']
            if 'nutau' in daughter:
                mothers = []
            elif 'nue' in daughter:
                mothers.extend(['K0S', 'mu'+rcharge])
            else:
                mothers.extend(['mu'+lcharge])
        elif categ == 'pr':
            if 'nutau' in daughter:
                mothers = ['D'+rcharge, 'Ds'+rcharge]
            else:
                mothers = ['D'+rcharge, 'Ds'+rcharge, 'D0'+rbar]#, 'Lambda0'+lbar]#, 'LambdaC+'+bar]
        else:
            mothers = [categ,]
        return mothers


    @staticmethod
    def esamp(enu, accuracy):
        """ returns the sampling of parent energies for a given enu
        """
        # TODO: replace 1e8 with MMC-prpl interpolated bounds
        return np.logspace(np.log10(enu),
                           np.log10(enu+1e8), 1000*accuracy)


    @staticmethod
    def projectiles():
        """Get allowed pimaries"""
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


    @staticmethod
    def nbody(fpath, esamp, enu, fn, l_ice):
        with np.load(fpath) as dfile:
            xmus = centers(dfile['xedges'])
            xnus = np.concatenate([xmus, [1]])
            vals = np.nan_to_num(dfile['histograms'])

            ddec = interpolate.RegularGridInterpolator((xnus, xmus), vals,
                                                       bounds_error=False, fill_value=None)
            emu_mat = xmus[:,None]*esamp[None,:]*Units.GeV
            pmu_mat = ddec(np.stack(np.meshgrid(enu/esamp, xmus), axis=-1))
            return 1-np.sum(pmu_mat*fn.prpl(np.stack([emu_mat, np.ones(emu_mat.shape)*l_ice], axis=-1)), axis=0)


    def get_solution(self,
                     particle_name,
                     grid_sol,
                     mag=0.,
                     grid_idx=None):
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

        # MCEq index conversion
        ref = self.mceq.pname2pref
        p_pdg = ParticleProperties.pdg_id[particle_name]
        reduce_res = True

        if grid_idx is None: # Surface only case
            sol = np.array([grid_sol[-1]])
            xv = np.array([self.X_vec[-1]])
        elif isinstance(grid_idx, bool) and not grid_idx: # Whole solution case
            sol = np.asarray(grid_sol)
            xv = np.asarray(self.X_vec)
            reduce_res = False
        elif grid_idx >= len(self.mceq.grid_sol): # Surface only case
            sol = np.array([grid_sol[-1]])
            xv = np.array([self.X_vec[-1]])
        else: # Particular height case
            sol = np.array([grid_sol[grid_idx]])
            xv = np.array([self.X_vec[grid_idx]])

        # MCEq solution for particle
        direct = sol[:,ref[particle_name].lidx():
                     ref[particle_name].uidx()]
        res = np.zeros(direct.shape)
        rho_air = self.mceq.density_model.X2rho(xv)

        # meson decay length
        decayl = ((self.mceq.e_grid * Units.GeV)
                  / ParticleProperties.mass_dict[particle_name]
                  * ParticleProperties.lifetime_dict[particle_name]
                  / Units.cm)

        # number of targets per cm2
        ndens = rho_air*Units.Na/Units.mol_air
        for prim in self.projectiles():
            prim_flux = sol[:,ref[prim].lidx():
                            ref[prim].uidx()]
            prim_xs = self.mceq.cs.get_cs(ParticleProperties.pdg_id[prim])
            try:
                int_yields = self.mceq.y.get_y_matrix(
                    ParticleProperties.pdg_id[prim],
                    p_pdg)
                res += np.sum(int_yields[None,:,:]*prim_flux[:,None,:]*prim_xs[None,None,:]*ndens[:,None,None], axis=2)
            except KeyError as e:
                continue

        res *= decayl[None,:]
        # combine with direct
        res[direct != 0] = direct[direct != 0]

        if particle_name[:-1] == 'mu':
            for _ in ['k_'+particle_name, 'pi_'+particle_name, 'pr_'+particle_name]:
                res += sol[:,ref[_].lidx():
                           ref[_].uidx()]

        res *= self.mceq.e_grid[None,:] ** mag

        if reduce_res:
            res = res[0]
        return res


    def get_rescale_phi(self, mother, grid_sol):
        """Flux of the mother at all heights"""
        dX = self.dX_vec*Units.gr/Units.cm**2
        rho = self.mceq.density_model.X2rho(self.X_vec)*Units.gr/Units.cm**3
        inv_decay_length_array = (ParticleProperties.mass_dict[mother] / (self.mceq.e_grid[:,None] * Units.GeV)) / (ParticleProperties.lifetime_dict[mother]*rho[None,:])
        rescale_phi = dX[None,:]* inv_decay_length_array * self.get_solution(mother, grid_sol, grid_idx=False).T
        return rescale_phi


    @lru_cache(2**12)
    def psib(self, mother, enu, accuracy, prpl):
        """ returns the suppression factor due to the sibling muon
        """
        l_ice = self.geom.overburden(self.costh)
        esamp = self.esamp(enu, accuracy)
        fn = MuonProb(prpl)
        if mother in ['D0', 'D0-bar']:
            reaching = nuVeto.nbody(
                resource_filename(
                'nuVeto','data/decay_distributions/D0_numu.npz'),
                esamp, enu, fn, l_ice)
        elif mother in ['D+', 'D-']:
            reaching = nuVeto.nbody(
                resource_filename(
                'nuVeto','data/decay_distributions/D+_numu.npz'),
                esamp, enu, fn, l_ice)
        elif mother in ['Ds+', 'Ds-']:
            reaching = nuVeto.nbody(
                resource_filename(
                'nuVeto','data/decay_distributions/Ds_numu.npz'),
                esamp, enu, fn, l_ice)
        elif mother == 'K0L':
            reaching = nuVeto.nbody(
                resource_filename(
                'nuVeto','data/decay_distributions/K0L_numu.npz'),
                esamp, enu, fn, l_ice)
        else:
            # Assuming muon energy is E_parent - E_nu
            reaching = 1. - fn.prpl(zip((esamp-enu)*Units.GeV,
                                    [l_ice]*len(esamp)))
        return reaching

            
    @lru_cache(2**12)
    def get_dNdEE(self, mother, daughter):
        """Differential parent-->neutrino (mother--daughter) yield"""
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


    @lru_cache(maxsize=2**12)
    def grid_sol(self, ecr=None, particle=None):
        """MCEq grid solution for \\frac{dN_{CR,p}}_{dE_p}"""
        if ecr is not None:
            self.mceq.set_single_primary_particle(ecr, particle)
        else:
            self.mceq.set_primary_model(*self.pmodel)
        self.mceq.solve(int_grid=self.X_vec, grid_var="X")
        return self.mceq.grid_sol


    @lru_cache(maxsize=2**12)
    def prob_nomu(self, ecr, particle, prpl='ice_allm97_step_1'):
        """Poisson probability of getting no muons"""
        grid_sol = self.grid_sol(ecr, particle)
        l_ice = self.geom.overburden(self.costh)
        mu = self.get_solution('mu-', grid_sol) + self.get_solution('mu+', grid_sol)

        fn = MuonProb(prpl)
        coords = zip(self.mceq.e_grid*Units.GeV, [l_ice]*len(self.mceq.e_grid))
        return np.exp(-np.trapz(mu*fn.prpl(coords),
                                self.mceq.e_grid))


    @lru_cache(2**12)
    def get_integrand(self, categ, daughter, enu, accuracy, prpl, ecr=None, particle=None):
        """flux*yield"""
        grid_sol = self.grid_sol(ecr, particle) # MCEq solution (fluxes tabulated as a function of height)
        esamp = self.esamp(enu, accuracy)
        mothers = self.categ_to_mothers(categ, daughter)
        nums = np.zeros((len(esamp),len(self.X_vec)))
        dens = np.zeros((len(esamp),len(self.X_vec)))
        for mother in mothers:
            dNdEE = self.get_dNdEE(mother, daughter)[-1]
            rescale_phi = self.get_rescale_phi(mother, grid_sol)
            rescale_phi = np.array([interpolate.interp1d(self.mceq.e_grid, rescale_phi[:,i], kind='quadratic', fill_value='extrapolate')(esamp) for i in xrange(rescale_phi.shape[1])]).T
            if 'numu' in daughter:
                # muon accompanies numu only
                pnmsib = self.psib(mother, enu, accuracy, prpl)
            else:
                pnmsib = np.ones(len(esamp))
            dnde = dNdEE(enu/esamp)/esamp
            nums += (dnde * pnmsib)[:,None]*rescale_phi
            dens += (dnde)[:,None]*rescale_phi

        return nums, dens


    def get_fluxes(self, enu, kind='conv_numu', accuracy=3, prpl='ice_allm97_step_1', corr_only=False):
        """Returns the flux and passing fraction
        for a particular neutrino energy, flux, and p_light
        """
        # prpl = probability of reaching * probability of light
        # prpl -> None ==> median for muon reaching
        categ, daughter = kind.split('_')

        esamp = self.esamp(enu, accuracy)

        # Correlated only (no need for the unified calculation here) [really just for testing]
        passed = 0
        total = 0
        if corr_only:
            # sum performs the dX integral
            nums, dens = self.get_integrand(categ, daughter, enu, accuracy, prpl)
            num = np.sum(nums, axis=1)
            den = np.sum(dens, axis=1)
            passed = integrate.trapz(num, esamp)
            total = integrate.trapz(den, esamp)
            return passed, total
                
        pmodel = self.pmodel[0](self.pmodel[1])

        #loop over primary particles
        for particle in pmodel.nucleus_ids:
            # A continuous input energy range is allowed between
            # :math:`50*A~ \\text{GeV} < E_\\text{nucleus} < 10^{10}*A \\text{GeV}`.

            # ecrs --> Energy of cosmic ray primaries
            # amu --> atomic mass of primary

            # evaluation points in E_CR
            ecrs = amu(particle)*np.logspace(2, 10, 10*accuracy)

            # pnm --> probability of no muon (just a poisson probability)
            pnm = [self.prob_nomu(ecr, particle, prpl) for ecr in ecrs]

            # pnmfn --> fine grid interpolation of pnm
            pnmfn = interpolate.interp1d(ecrs, pnm, kind='cubic',
                                         assume_sorted=True, bounds_error=False,
                                         fill_value=(1,np.nan))
            # nums --> numerator
            nums = []
            # dens --> denominator
            dens = []
            # istart --> integration starting point, the lowest energy index for the integral
            istart = max(0, np.argmax(ecrs > enu) - 1)
            for ecr in ecrs[istart:]: # integral in primary energy (E_CR)
                # cr_flux --> cosmic ray flux
                # phim2 --> units of flux * m^2 (look it up in the units)
                cr_flux = pmodel.nucleus_flux(particle, ecr.item())*Units.phim2
                # poisson exp(-Nmu) [last term in eq 12]
                pnmarr = pnmfn(ecr-esamp)
                # cubic splining doesn't enforce 0-1 bounds
                pnmarr[pnmarr>1] = 1
                pnmarr[pnmarr<0] = 0
                # print pnmarr

                num_ecr = 0 # single entry in nums
                den_ecr = 0 # single entry in dens

                # dEp
                # integral in Ep
                nums_ecr, dens_ecr = self.get_integrand(categ, daughter, enu, accuracy, prpl, ecr, particle)
                num_ecr = integrate.trapz(np.sum(nums_ecr, axis=1)*pnmarr, esamp)
                den_ecr = integrate.trapz(np.sum(dens_ecr, axis=1), esamp)

                nums.append(num_ecr*cr_flux/Units.phicm2)
                dens.append(den_ecr*cr_flux/Units.phicm2)
            # dEcr
            passed += integrate.trapz(nums, ecrs[istart:])
            total += integrate.trapz(dens, ecrs[istart:])

        return passed, total


@lru_cache(maxsize=2**12)
def builder(cos_theta, pmodel, hadr, barr_mods, depth, density):
    return nuVeto(cos_theta, pmodel, hadr, barr_mods, depth, density)


def passing(enu, cos_theta, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', barr_mods=(), depth=1950*Units.m, density=('CORSIKA', ('SouthPole', 'June')), accuracy=3, fraction=True, prpl='ice_allm97_step_1', corr_only=False):
    sv = builder(cos_theta, pmodel, hadr, barr_mods, depth, density)
    num, den = sv.get_fluxes(enu, kind, accuracy, prpl, corr_only)
    return num/den if fraction else num


def fluxes(enu, cos_theta, kind='conv_numu', pmodel=(pm.HillasGaisser2012, 'H3a'), hadr='SIBYLL2.3c', barr_mods=(), depth=1950*Units.m, density=('CORSIKA', ('SouthPole', 'June')), accuracy=3, prpl='ice_allm97_step_1', corr_only=False):
    sv = builder(cos_theta, pmodel, hadr, barr_mods, depth, density)
    return sv.get_fluxes(enu, kind, accuracy, prpl, corr_only)
