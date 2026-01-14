"""nuVeto calculation [ref. JCAP07(2018)047]

This module computes the probability that an atmospheric neutrino will be
accompanied by a sibling muon produced in the same cosmic ray airshower at a
given depth.


"""

from functools import lru_cache
from importlib.resources import files
import logging
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

from MCEq.core import MCEqRun
from MCEq import config
import crflux.models as pm
from .utils import Units, ParticleProperties, Geometry, amu, centers
from .mu import MuonProb
from .uncertainties import BARR, barr_unc

logger = logging.getLogger(__name__)

class nuVeto(object):
    """Class for computing the neutrino passing fraction i.e. (1-(Veto probability))
    Initializes the nuVeto object for a specific physical configuration.

    Parameters
    ----------
    costh : float
        Cos(theta), the cosine of the neutrino zenith at the detector.
    pmodel : tuple(CR model class, arguments)
        CR Flux from `crflux.models.pm`.
    hadr : str
        Hadronic interaction model.
    barr_mods : dict or list
        Barr parameters.
    depth : float
        Depth below the surface with units attached (e.g., val*Units.m).
    density : tuple
        Atmospheric density specifier for MCEq.
    debug_level : int
        MCEq debug level.

    Notes
    -----
    A separate MCEq instance needs to be created for each combination of the
    passed arguments. Available fluxes are documented in `crflux.models.pm`.
    """
    def __init__(
        self,
        costh,
        pmodel=(pm.HillasGaisser2012, "H3a"),
        hadr="SIBYLL2.3c",
        barr_mods=(),
        depth=1950 * Units.m,
        density=("CORSIKA", ("SouthPole", "December")),
        debug_level=1,
    ):
        self.costh = costh
        self.pmodel = pmodel
        self.geom = Geometry(depth)
        theta = np.degrees(np.arccos(self.geom.cos_theta_eff(self.costh)))
        if density[0] == "MSIS00_IC":
            logger.info(
                'Passing "MSIS00_IC" assumes IceCube-centered coordinates, '
                'which obviates the depth used here. Switching to "MSIS00" '
                "for identical results."
            )
            density = ("MSIS00", density[1])

        config.debug_level = debug_level
        # config.enable_em = False
        config.enable_muon_energy_loss = False
        config.return_as = "total energy"
        config.adv_set["allowed_projectiles"] = [
            2212,
            -2212,
            2112,
            -2112,
            211,
            -211,
            321,
            -321,
            3122,
            -3122,
            310,
            130,
        ]
        self.mceq = MCEqRun(
            # provide the string of the interaction model
            interaction_model=hadr,
            # primary cosmic ray flux model
            # support a tuple (primary model class (not instance!), arguments)
            primary_model=pmodel,
            # zenith angle \theta in degrees, measured positively from vertical direction at surface
            theta_deg=theta,
            # atmospheric density model
            density_model=density,
        )

        if len(barr_mods) > 0:
            for barr_mod in barr_mods:
                # Modify proton-air -> mod[0]
                self.mceq.set_mod_pprod(2212, BARR[barr_mod[0]].pdg, barr_unc, barr_mod)
            # Populate the modifications to the matrices by re-filling the interaction matrix
            self.mceq.regenerate_matrices(skip_decay_matrix=True)

        X_vec = np.logspace(np.log10(2e-3), np.log10(self.mceq.density_model.max_X), 12)
        self.dX_vec = np.diff(X_vec)
        self.X_vec = 10 ** centers(np.log10(X_vec))

    @staticmethod
    def categ_to_mothers(categ, daughter):
        """Get the parents for this category"""
        rcharge = "-" if "bar" in daughter else "+"
        lcharge = "+" if "bar" in daughter else "-"
        rbar = "bar" if "bar" in daughter else ""
        if categ == "conv":
            if "nu_tau" in daughter:
                return []

            mothers = [f"pi{rcharge}", f"K{rcharge}", "K_L0"]
            if "nu_e" in daughter:
                mothers.extend(["K_S0", f"mu{rcharge}"])
            elif "nu_mu" in daughter:
                mothers.extend([f"mu{lcharge}"])
            return mothers
        if categ == "pr":
            if "nu_tau" in daughter:
                return [f"D{rcharge}", f"D_s{rcharge}"]
            # , 'Lambda'+lbar+'0']#, 'Lambda_c'+rcharge]
            return [f"D{rcharge}", f"D_s{rcharge}", f"D{rbar}0"]
        if categ == "total":
            return nuVeto.categ_to_mothers(
                "conv", daughter
            ) + nuVeto.categ_to_mothers("pr", daughter)

        return [categ,]

    @staticmethod
    def esamp(enu, accuracy, emu_max=1.e8):
        """returns the sampling of parent energies for a given enu

        The sampled parent energy cannot exceed enu+emu_max, as then the decay-muon energy
        can exceed the energy range over which MMC data was tabulated
        """
        if not np.isfinite(emu_max):
            logger.warning("The passed emu_max is not finite, assuming 1.e8 for parent-energy sampling.")
            emu_max = 1.e8
        return np.logspace(np.log10(enu), np.log10(enu+emu_max), int(1000 * accuracy))

    @staticmethod
    def projectiles():
        """Get allowed projectiles"""
        pdg_ids = config.adv_set["allowed_projectiles"]
        return [ParticleProperties.modtab.pdg2modname[_] for _ in pdg_ids]

    @staticmethod
    def nbody(fpath, esamp, enu, fn, l_ice):
        """Rely on tabulated decay kinematics for n>2body decays. Returns 1-Pdet."""
        with fpath.open("rb") as dfile:
            data = np.load(dfile)
            xmus = centers(data["xedges"])
            xnus = np.concatenate([xmus, [1]])
            vals = np.nan_to_num(data["histograms"])

        ddec = interpolate.RegularGridInterpolator(
            (xnus, xmus), vals, bounds_error=False, fill_value=None
        )
        emu_mat = xmus[:, None] * esamp[None, :] * Units.GeV
        pmu_mat = ddec(np.stack(np.meshgrid(enu / esamp, xmus), axis=-1))
        return np.maximum(0.0, 1 - np.sum(
            pmu_mat
            * fn.prpl(np.stack([emu_mat, np.ones(emu_mat.shape) * l_ice], axis=-1)),
            axis=0,
        ))

    @staticmethod
    @lru_cache(2**12)
    def psib(l_ice, mother, enu, accuracy, prpl):
        """returns the suppression factor due to the sibling muon"""
        fn = MuonProb(prpl)
        esamp = nuVeto.esamp(enu, accuracy, fn.eis[-1])
        if mother in ["D0", "D0-bar"]:
            return nuVeto.nbody(
                files("nuVeto") / "data" / "decay_distributions" / "D0_numu.npz",
                esamp,
                enu,
                fn,
                l_ice,
            )
        if mother in ["D+", "D-"]:
            return nuVeto.nbody(
                files("nuVeto") / "data" / "decay_distributions" / "D+_numu.npz",
                esamp,
                enu,
                fn,
                l_ice,
            )
        if mother in ["Ds+", "Ds-"]:
            return nuVeto.nbody(
                files("nuVeto") / "data" / "decay_distributions" / "Ds_numu.npz",
                esamp,
                enu,
                fn,
                l_ice,
            )
        if mother == "K0L":
            return nuVeto.nbody(
                files("nuVeto") / "data" / "decay_distributions" / "K0L_numu.npz",
                esamp,
                enu,
                fn,
                l_ice,
            )
        if mother in ["mu+", "mu-"]:
            return np.ones_like(esamp)

        # Assuming muon energy is E_parent - E_nu
        return 1.0 - fn.prpl(
            list(zip((esamp - enu) * Units.GeV, [l_ice] * len(esamp)))
        )

    @lru_cache(maxsize=2**12)
    def get_dNdEE(self, mother, daughter):
        """Differential parent-->neutrino (mother--daughter) yield"""
        ihijo = 20
        e_grid = self.mceq.e_grid
        delta = self.mceq.e_widths
        x_range = e_grid[ihijo] / e_grid
        rr = ParticleProperties.rr(mother, daughter)
        dNdEE_edge = ParticleProperties.br_2body(mother, daughter) / (1 - rr)
        dN_mat = self.mceq._decays.get_matrix(
            (ParticleProperties.pdg_id[mother], 0),
            (ParticleProperties.pdg_id[daughter], 0),
        )
        dNdEE = dN_mat[ihijo] * e_grid / delta
        logx = np.log10(x_range)
        logx_width = -np.diff(logx)[0]
        good = (logx + logx_width / 2 < np.log10(1 - rr)) & (x_range >= 5.0e-2)

        x_low = x_range[x_range < 5e-2]
        dNdEE_low = np.array([dNdEE[good][-1]] * x_low.size)

        def dNdEE_interp(x_):
            return interpolate.pchip(
                np.concatenate([[1 - rr], x_range[good], x_low])[::-1],
                np.concatenate([[dNdEE_edge], dNdEE[good], dNdEE_low])[::-1],
                extrapolate=True,
            )(x_) * np.heaviside(1 - rr - x_, 1)

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
    def nmu(self, ecr, particle, prpl="ice_allm97_step_1"):
        """Number of expected muons for a given primary energy / particle.
        Used to compute the Poisson probability of getting no muons"""
        grid_sol = self.grid_sol(ecr, particle)
        l_ice = self.geom.overburden(self.costh)
        # np.abs hack to prevent negative fluxes
        mu = np.abs(self.get_solution("mu-", grid_sol)) + np.abs(
            self.get_solution("mu+", grid_sol)
        )
        fn = MuonProb(prpl)
        coords = list(
            zip(self.mceq.e_grid * Units.GeV, [l_ice] * len(self.mceq.e_grid))
        )

        return integrate.trapezoid(
            mu * fn.prpl(coords) * self.mceq.e_grid, np.log(self.mceq.e_grid)
        )

    @lru_cache(maxsize=2**12)
    def get_rescale_phi(self, mother, ecr=None, particle=None):
        """Flux of the mother at all heights"""
        grid_sol = self.grid_sol(
            ecr, particle
        )  # MCEq solution (fluxes tabulated as a function of height)
        dX = self.dX_vec * Units.gr / Units.cm**2
        rho = self.mceq.density_model.X2rho(self.X_vec) * Units.gr / Units.cm**3
        inv_decay_length_array = (
            ParticleProperties.mass_dict[mother]
            / (self.mceq.e_grid[:, None] * Units.GeV)
        ) / (ParticleProperties.lifetime_dict[mother] * rho[None, :])
        rescale_phi = (
            dX[None, :]
            * inv_decay_length_array
            * self.get_solution(mother, grid_sol, grid_idx=False).T
        )
        return rescale_phi

    def get_integrand(
        self, categ, daughter, enu, accuracy, prpl, ecr=None, particle=None
    ):
        """flux*yield"""
        esamp = self.esamp(enu, accuracy, MuonProb(prpl).eis[-1])
        mothers = self.categ_to_mothers(categ, daughter)
        nums = np.zeros((len(esamp), len(self.X_vec)))
        dens = np.zeros((len(esamp), len(self.X_vec)))
        for mother in mothers:
            dNdEE = self.get_dNdEE(mother, daughter)[-1]
            rescale_phi = self.get_rescale_phi(mother, ecr, particle)

            ###
            # TODO: optimize to only run when esamp[0] is non-zero
            rescale_phi = np.exp(
                np.array(
                    [
                        interpolate.interp1d(
                            np.log(self.mceq.e_grid[rescale_phi[:, i] > 0]),
                            np.log(rescale_phi[:, i][rescale_phi[:, i] > 0]),
                            kind="quadratic",
                            bounds_error=False,
                            fill_value=-np.inf,
                        )(np.log(esamp))
                        if np.count_nonzero(rescale_phi[:, i] > 0) > 2
                        else [
                            -np.inf,
                        ]
                        * esamp.shape[0]
                        for i in range(rescale_phi.shape[1])
                    ]
                )
            ).T

            if "nu_mu" in daughter:
                # muon accompanies nu_mu only
                pnmsib = self.psib(
                    self.geom.overburden(self.costh), mother, enu, accuracy, prpl
                )
            else:
                pnmsib = np.ones_like(esamp)
            dnde = dNdEE(enu / esamp) / esamp
            nums += (dnde * pnmsib)[:, None] * rescale_phi
            dens += (dnde)[:, None] * rescale_phi

        return nums, dens

    def get_solution(self, particle_name, grid_sol, mag=0.0, grid_idx=None):
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
        ref = self.mceq.pman.pname2pref
        p_pdg = ParticleProperties.pdg_id[particle_name]
        reduce_res = True

        if grid_idx is None:  # Surface only case
            sol = np.array([grid_sol[-1]])
            xv = np.array([self.X_vec[-1]])
        elif isinstance(grid_idx, bool) and not grid_idx:  # Whole solution case
            sol = np.asarray(grid_sol)
            xv = np.asarray(self.X_vec)
            reduce_res = False
        elif grid_idx >= len(self.mceq.grid_sol):  # Surface only case
            sol = np.array([grid_sol[-1]])
            xv = np.array([self.X_vec[-1]])
        else:  # Particular height case
            sol = np.array([grid_sol[grid_idx]])
            xv = np.array([self.X_vec[grid_idx]])

        # MCEq solution for particle
        direct = sol[:, ref[particle_name].lidx : ref[particle_name].uidx]
        res = np.zeros(direct.shape)
        rho_air = 1.0 / self.mceq.density_model.r_X2rho(xv)

        # meson decay length
        decayl = (
            (self.mceq.e_grid * Units.GeV)
            / ParticleProperties.mass_dict[particle_name]
            * ParticleProperties.lifetime_dict[particle_name]
            / Units.cm
        )

        # number of targets per cm2
        ndens = rho_air * Units.Na / config.A_target
        sec = self.mceq.pman[p_pdg]
        prim2mceq = {
            "p+-bar": "pbar-",
            "n0-bar": "nbar0",
            "D0-bar": "Dbar0",
            "Lambda0-bar": "Lambdabar0",
        }
        for prim in self.projectiles():
            if prim in prim2mceq:
                _ = prim2mceq[prim]
            else:
                _ = prim
            prim_flux = sol[:, ref[_].lidx : ref[_].uidx]
            proj = self.mceq.pman[ParticleProperties.pdg_id[prim]]
            prim_xs = proj.inel_cross_section()
            try:
                int_yields = proj.hadr_yields[sec]
                res += np.sum(
                    int_yields[None, :, :]
                    * prim_flux[:, None, :]
                    * prim_xs[None, None, :]
                    * ndens[:, None, None],
                    axis=2,
                )
            except KeyError:
                continue

        res *= decayl[None, :]
        # combine with direct
        res[direct != 0] = direct[direct != 0]

        if particle_name[:-1] == "mu":
            for _ in [f"k_{particle_name}", f"pi_{particle_name}"]:
                res += sol[:, ref[f"{_}_l"].lidx : ref[f"{_}_l"].uidx]
                res += sol[:, ref[f"{_}_r"].lidx : ref[f"{_}_r"].uidx]

        res *= self.mceq.e_grid[None, :] ** mag

        if reduce_res:
            res = res[0]
        return res

    def get_fluxes(
        self,
        enu,
        kind="conv nu_mu",
        accuracy=3.5,
        prpl="ice_allm97_step_1",
        corr_only=False,
    ):
        """Returns the flux and passing fraction
        for a particular neutrino energy, flux, and p_light
        """
        # prpl = probability of reaching * probability of light
        # prpl -> None ==> median for muon reaching
        categ, daughter = kind.split()

        esamp = self.esamp(enu, accuracy, MuonProb(prpl).eis[-1])

        # Correlated only (no need for the unified calculation here) [really just for testing]
        passed = 0
        total = 0
        if corr_only:
            # sum performs the dX integral
            nums, dens = self.get_integrand(categ, daughter, enu, accuracy, prpl)
            num = np.sum(nums, axis=1)
            den = np.sum(dens, axis=1)
            passed = integrate.trapezoid(num, esamp)
            total = integrate.trapezoid(den, esamp)
            return passed, total

        pmodel = self.pmodel[0](self.pmodel[1])

        # loop over primary particles
        for particle in pmodel.nucleus_ids:
            # A continuous input energy range is allowed between
            # :math:`50*A~ \\text{GeV} < E_\\text{nucleus} < 10^{10}*A \\text{GeV}`.

            # ecrs --> Energy of cosmic ray primaries
            # amu --> atomic mass of primary

            # evaluation points in E_CR
            ecrs = amu(particle) * np.logspace(2, 10, int(10 * accuracy))

            # pnm (exp(-nmu)) --> probability of no muon (just a poisson probability)
            nmu = [self.nmu(ecr, particle, prpl) for ecr in ecrs]

            # nmufn --> fine grid interpolation of pnm
            nmufn = interpolate.interp1d(
                ecrs,
                nmu,
                kind="linear",
                assume_sorted=True,
                bounds_error=False,
                fill_value=(0, np.nan),
            )
            # nums --> numerator
            nums = []
            # dens --> denominator
            dens = []
            # istart --> integration starting point, the lowest energy index for the integral
            istart = max(0, np.argmax(ecrs > enu) - 1)
            for ecr in ecrs[istart:]:  # integral in primary energy (E_CR)
                # cr_flux --> cosmic ray flux
                # phim2 --> units of flux * m^2 (look it up in the units)
                cr_flux = pmodel.nucleus_flux(particle, ecr.item()) * Units.phim2
                # poisson exp(-Nmu) [last term in eq 12]
                pnmarr = np.exp(-nmufn(ecr - esamp))

                num_ecr = 0  # single entry in nums
                den_ecr = 0  # single entry in dens

                # dEp
                # integral in Ep
                nums_ecr, dens_ecr = self.get_integrand(
                    categ, daughter, enu, accuracy, prpl, ecr, particle
                )
                num_ecr = integrate.trapezoid(np.sum(nums_ecr, axis=1) * pnmarr, esamp)
                den_ecr = integrate.trapezoid(np.sum(dens_ecr, axis=1), esamp)

                nums.append(num_ecr * cr_flux / Units.phicm2)
                dens.append(den_ecr * cr_flux / Units.phicm2)
            # dEcr
            passed += integrate.trapezoid(nums, ecrs[istart:])
            total += integrate.trapezoid(dens, ecrs[istart:])

        return passed, total


@lru_cache(maxsize=2**12)
def builder(cos_theta, pmodel, hadr, barr_mods, depth, density):
    """
    Creates and caches (LRU) a nuVeto object for the given parameters.

    Parameters
    ----------
    costh : float
        Cos(theta), the cosine of the neutrino zenith at the detector.
    pmodel : tuple(CR model class, arguments)
        CR Flux from crflux.models.
    hadr : str
        Hadronic interaction model.
    barr_mods : dict or list
        Barr parameters.
    depth : float
        Depth below the surface with units attached (e.g. val*Units.m).
    density : tuple
        Atmospheric density specifier for MCEq.

    Returns
    -------
    nuVeto
        A cached nuVeto object corresponding to the provided arguments.

    Notes
    -----
    A separate MCEq instance needs to be created for each combination of the
    passed arguments. Available fluxes can be found in `crflux.models.pm`.
    """
    return nuVeto(cos_theta, pmodel, hadr, barr_mods, depth, density)


def passing(
    enu,
    cos_theta,
    kind="conv nu_mu",
    pmodel=(pm.HillasGaisser2012, "H3a"),
    hadr="SIBYLL2.3c",
    barr_mods=(),
    depth=1950 * Units.m,
    density=("CORSIKA", ("SouthPole", "December")),
    accuracy=3.5,
    fraction=True,
    prpl="ice_allm97_step_1",
    corr_only=False,
):
    """
    Returns the passing atmospheric neutrino flux or passing fraction.

    Parameters
    ----------
    enu : float
        Neutrino energy.
    costh : float
        Cos(theta), the cosine of the neutrino zenith in detector coordinates.
    kind : str
        Specifier for what type of atmos. nu to assume, can be 
        '(conv|pr|_parent_) nu_(e|mu)(bar)'.
    pmodel : tuple(CR model class, arguments)
        CR Flux from crflux.models.
    hadr : str
        Hadronic interaction model.
    barr_mods : dict or list
        Barr parameters.
    depth : float
        Depth below the surface with units attached (e.g. val*Units.m).
    density : tuple
        Atmospheric density specifier for MCEq.
    accuracy : float
        Higher values will increase density of parent-energy sampling.
    fraction : bool
        If True, returns the passing fraction, else returns the passing flux.
    prpl : str, RegularGridInterpolator, or None
        The muon detection probability, can be string filename stem, 
        an object, or if None will use median approximation.
    corr_only : bool
        Whether or not to include the uncorrelated muons contribution.

    Returns
    -------
    float
        The passing atmospheric neutrino flux or the passing fraction.
    """
    sv = builder(cos_theta, pmodel, hadr, barr_mods, depth, density)
    num, den = sv.get_fluxes(enu, kind, accuracy, prpl, corr_only)
    return num / den if fraction else num


def fluxes(
    enu,
    cos_theta,
    kind="conv nu_mu",
    pmodel=(pm.HillasGaisser2012, "H3a"),
    hadr="SIBYLL2.3c",
    barr_mods=(),
    depth=1950 * Units.m,
    density=("CORSIKA", ("SouthPole", "December")),
    accuracy=3.5,
    prpl="ice_allm97_step_1",
    corr_only=False,
):
    """
    Returns passing and total atmospheric neutrino fluxes.

    Parameters
    ----------
    enu : float
        Neutrino energy.
    costh : float
        Cos(theta), the cosine of the neutrino zenith in detector coordinates.
    kind : str
        Specifier for what type of atmos. nu to assume, can be 
        '(conv|pr|_parent_) nu_(e|mu)(bar)'.
    pmodel : tuple(CR model class, arguments)
        CR Flux from crflux.models.
    hadr : str
        Hadronic interaction model.
    barr_mods : dict or list
        Barr parameters.
    depth : float
        Depth below the surface with units attached (e.g. val*Units.m).
    density : tuple
        Atmospheric density specifier for MCEq.
    accuracy : float
        Higher values will increase density of parent-energy sampling.
    prpl : str, RegularGridInterpolator, or None
        The muon detection probability, can be string filename stem, 
        an object, or if None will use median approximation.
    corr_only : bool
        Whether or not to include the uncorrelated muons contribution.

    Returns
    -------
    tuple of float
        The passing and total atmospheric neutrino fluxes.
    """
    sv = builder(cos_theta, pmodel, hadr, barr_mods, depth, density)
    return sv.get_fluxes(enu, kind, accuracy, prpl, corr_only)
