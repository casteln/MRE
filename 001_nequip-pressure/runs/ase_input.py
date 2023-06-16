import numpy as np
import logging
import time
import torch
import sys
from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.velocitydistribution import Stationary, ZeroRotation

from nequip.ase import NequIPCalculator
from nequip.ase import NoseHoover




# adapted from https://github.com/mir-group/nequip/blob/main/nequip/scripts/run_md.py

def save_to_xyz(atoms, prefix):
    """
    Save structure to extended xyz file.
    :param atoms: ase.Atoms object to save
    :param logdir, str, path/to/logging/directory
    :param prefix: str, prefix to use for storing xyz files
    """
    write(
        filename=prefix + ".xyz",
        images=atoms,
        format="extxyz",
        append=True,
    )

def write_ase_md_config(curr_atoms, curr_step, dt):
    """Write time, positions, forces, and atomic kinetic energies to log file.
    :param curr_atoms: ase.Atoms object, current system to log
    :param curr_step: int, current step / frame in MD simulation
    :param dt: float, MD time step
    """
    parsed_temperature = curr_atoms.get_temperature()
    parsed_potentialenergy = curr_atoms.get_potential_energy()
    parsed_totalenergy = curr_atoms.get_total_energy()

    # frame
    log_txt = "Frame: {}".format(str(curr_step))
    try:
        log_txt += "\t SimulationTime: {:.6f}\t Temperature: {:.8f} K\t PotentialEnergy: {:.8f} eV\t TotalEnergy: {:.8f} eV".format(
        dt * curr_step, parsed_temperature, parsed_potentialenergy, parsed_totalenergy
    )
    except TypeError:
        log_txt += "\t SimulationTime: {:.6f}\t Temperature: {:.8f} K\t PotentialEnergy: {:.8f} eV\t TotalEnergy: {:.8f} eV".format(
        dt * curr_step, parsed_temperature, parsed_potentialenergy.item(), parsed_totalenergy.item()
    )

    logging.info(log_txt)

def run_md(initial_xyz,model,temperature,dt,nvt_q,n_steps,out):

    logfilename = out+'_ase_md_run_'+str(time.time())+'.log'

    np.random.seed(666)
    torch.manual_seed(666)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Found device:", device)

    logging.basicConfig(filename=logfilename, format="%(message)s", level=logging.INFO)

    # load atoms
    atoms = read(initial_xyz, index=0)

    # build nequip calculator
    calc = NequIPCalculator.from_deployed_model(
        model_path=model,
        device=device,
        properties=["energy", "stress"] # added this line to output stress
    )

    atoms.set_calculator(calc=calc)

    # set starting temperature
    MaxwellBoltzmannDistribution(atoms=atoms, temp=temperature * units.kB)

    ZeroRotation(atoms)
    Stationary(atoms)

    nvt_dyn = NoseHoover(
        atoms=atoms,
        timestep=dt * units.fs,
        temperature=temperature,
        nvt_q=nvt_q,
    )

    # log first frame
    logging.info(
        f"#Starting dynamics with Nose-Hoover Thermostat with nvt_q: {nvt_q}"
    )
    write_ase_md_config(curr_atoms=atoms, curr_step=0, dt=dt)

    save_to_xyz(atoms, out)

    for i in range(1, n_steps):
        nvt_dyn.run(steps=1)

        if not i % 10:
            write_ase_md_config(curr_atoms=atoms, curr_step=i, dt=dt)

        # append current structure to xyz file
        if not i % 10:
            save_to_xyz(atoms, out)

    # add final output
    save_to_xyz(atoms, out + "_final")

    print("finished...")


if __name__ == "__main__":
    # inital xyz file
    init = "atoms.xyz"

    # machine learned model from Nequip (*.pth)
    model = "model.pth"

    # temperature in kelvin
    temp = 300

    # time step in fs
    dt = 0.5

    # time constant for thermostat
    nvtq = 500

    # number of steps
    nsteps = 100000

    #output file name
    out = "output"

    run_md(init, model, temp, dt, nvtq, nsteps, out)