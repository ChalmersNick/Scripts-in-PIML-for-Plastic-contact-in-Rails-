import numpy as np

rrlo, rrhi = 250, 350
rwxlo, rwxhi = 700, 1000
rwylo, rwyhi = 980/2, 1300/2
loadlo, loadhi = 44*9.82e3/8/4, 21/8*9.82e3

n = 3

rhors = np.linspace(1/rrhi, 1/rrlo, n)
rhowxs = np.linspace(1/rwxhi, 1/rwxlo, n)
rhowys = np.linspace(1/rwyhi, 1/rwylo, n)
loads = -np.linspace(loadhi, loadlo, n)

rrs = 1/rhors
rwxs = 1/rhowxs
rwys = 1/rhowys

lines = ['Rail radius,Wheel radius x,Wheel radius y,Load\n']

for rr, rwx, rwy, load in zip(rrs, rwxs, rwys, loads):
    lines.append(f'{rr:.8f},{rwx:.8f},{rwy:.8f},{load:.6f}\n')

with open(
    # 'Simulation\\' \
        'inputdata.txt', 'w'
) as f:
    f.writelines(lines)