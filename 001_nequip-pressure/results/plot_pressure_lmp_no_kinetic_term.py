import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import atomman.lammps as lmp
import ase
import ase.io
import os

working_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_dir) # change working dir as where the python script is

# change to correct paths
path_to_lammps = '../output/lmp-null/log.run'
path_to_ase = '../output/ase/output.xyz'

# Read lammps

results = lmp.Log()
results.read(path_to_lammps)
df = results.simulations[0]['thermo']
df = df[:10000] # same length as ase
lmp_press = df['Press']
step = df['Step']

# no kinetic term

# code from https://github.com/mir-group/pair_nequip/blob/stress/tests/test_python_repro.py#L317-L344

# https://docs.lammps.org/compute_pressure.html
# > The ordering of values in the symmetric pressure tensor is as follows: pxx, pyy, pzz, pxy, pxz, pyz.
lmp_stress_no_kinetic = np.array(
    [
        [df['c_stress[1]'], df['c_stress[4]'], df['c_stress[5]']],
        [df['c_stress[4]'], df['c_stress[2]'], df['c_stress[6]']],
        [df['c_stress[5]'], df['c_stress[6]'], df['c_stress[3]']],
    ]
)
lmp_press_no_kinetic = np.trace(lmp_stress_no_kinetic) / 3


print("Average pressure with LAMMPS: %s", np.mean(lmp_press))
print("Average pressure with LAMMPS - no kinetic term: %s", np.mean(lmp_press_no_kinetic))

# Plot

plt.plot(step, lmp_press, label = 'LAMMPS')
plt.plot(step, lmp_press_no_kinetic, label = 'LAMMPS - No kinetic term')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Pressure (bar)')

plt.savefig('Pressure(step)_no_kinetic_term_lmp.pdf')

# Read ASE

traj = ase.io.read(path_to_ase, index=':')

from_ev_per_ang_3_to_bars = 1.602176634 * 1e6 # convert from ase units to bars (lmp units)
# multiply by -1 to have the same convention than lammps
stress = (-1) * from_ev_per_ang_3_to_bars * np.array([atom.info['stress'] for atom in traj])
ase_press = np.trace(stress, axis1 = 1, axis2 = 2) / 3

print("Average pressure with ASE: %s", np.mean(ase_press))

# Plot

plt.figure()
plt.plot(step, lmp_press_no_kinetic, label = 'LAMMPS - no kinetic term')
plt.plot(step, ase_press, label = 'ASE')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Pressure (bar)')

plt.savefig('Pressure(step)_no_kinetic_term.pdf')

