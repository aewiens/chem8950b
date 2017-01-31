#!/usr/bin/env python
import psi4, numpy as np
from collections import namedtuple

import configparser
config = configparser.ConfigParser()
config.read('Options.ini')

molecule = psi4.geometry(config['DEFAULT']['molecule'])
basis    = psi4.core.BasisSet.build(molecule, 'BASIS', 'STO-3G', puream=0)
mints    = psi4.core.MintsHelper(basis)
nShell   = basis.nshell() 
nbf      = basis.nao() 

recursionResults = namedtuple('recursionResults',['x','y','z'])


def osRecursion(PA, PB, alpha, AMa, AMb):
	if len(PA) != 3 or len(PB) != 3:
		raise ""

	"""
	##  Allocate x, y, and z matrices
	x = np.zeros((AMa+1, AMb+1))
	y = np.zeros((AMa+1, AMb+1))
	z = np.zeros((AMa+1, AMb+1))
	"""

	##  Perform recursion

	xyz = []
	coeff = 1.0

	for i in range(3):
		I = np.zeros((AMa+1, AMb+1))
		I[0,0] = 1.0

		for a in range(1,AMa+1):
			I[a,0] = PA[i]*I[a-1,0]
			if a >= 2:
				I[a,0] += (a-1)/(2*alpha)*I[a-2,0] 

		for b in range(1,AMb+1):
			I[0,b] = PB[i]*I[0,b-1]
			if b >= 2:
				I[0,b] += (b-1)/(2*alpha)*I[0,b-2]
		
		for a in range(1,AMa+1):
			for b in range(1,AMb+1):
				I[a,b] = PB[i]*I[a,b-1] + a/(2*alpha)*I[a-1,b-1] 
				if b >= 2:
					I[a,b] +=  (b-1)/(2*alpha)*I[a,b-2]

		xyz.append(I)
	return xyz

XYZ = osRecursion([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], 5.0, 3, 3)

print('x=\n', XYZ[0])
print('y=\n', XYZ[1])
print('z=\n', XYZ[2])

print( type(XYZ[0]))
