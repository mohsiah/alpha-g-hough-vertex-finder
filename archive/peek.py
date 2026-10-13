"""peek.py -- shows what the 5 columns of the ALPHA-g file are. Usage:  python peek.py test.h5"""
import sys
import numpy as np
import h5py

with h5py.File(sys.argv[1], 'r') as f:
    a = np.asarray(f['data'][:200], dtype=float)          # first 200 events only
print('array shape (events, rows, columns):', a.shape)

e = a[0]
print('\nEVENT 0, first 6 rows (each row = 5 numbers):')
print(np.round(e[:6], 2))
print('EVENT 0, rows 6-8 from the END:')
print(np.round(e[-3:], 2))
print('rows that are all zeros in event 0:', int(np.sum(np.all(e == 0, axis=1))), 'of', len(e))

nz = a[np.any(a != 0, axis=2)]                            # every non-zero row of the 200 events
print('\nOver all non-zero rows of 200 events, per column:')
for c in range(5):
    print(f'  column {c}: min {nz[:, c].min():9.2f}  max {nz[:, c].max():9.2f}  mean {nz[:, c].mean():9.2f}')

print('\nIs a column CONSTANT within each event? (spread inside event 0, non-zero rows)')
nz0 = e[np.any(e != 0, axis=1)]
for c in range(5):
    print(f'  column {c}: spread {np.ptp(nz0[:, c]):9.2f}')