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
path_to_lammps = '../output/lmp/log.run.1.equilibrate'
path_to_ase = '../output/ase/output.xyz'

# Read lammps

results = lmp.Log()
results.read(path_to_lammps)
df = results.simulations[0]['thermo']
lmp_press = df['Press']
step = df['Step']

print("Average pressure with LAMMPS: %s", np.mean(lmp_press))


# Read ASE

traj = ase.io.read(path_to_ase, index=':')

from_ev_per_ang_3_to_bars = 1.602176634 * 1e6 # convert from ase units to bars (lmp units)
# multiply by -1 to have the same convention than lammps
stress = (-1) * from_ev_per_ang_3_to_bars * np.array([atom.info['stress'] for atom in traj])
ase_press = np.trace(stress, axis1 = 1, axis2 = 2) / 3

print("Average pressure with ASE: %s", np.mean(ase_press))

# Plot

plt.plot(step, lmp_press, label = 'LAMMPS')
plt.plot(step[:-1], ase_press, label = 'ASE')
plt.legend()
plt.xlabel('Step')
plt.ylabel('Pressure (bar)')

plt.savefig('Pressure(step).pdf')

