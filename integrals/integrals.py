#!/usr/bin/env python
import psi4, numpy as np
import configparser
import sys
sys.path.append("../uhf/")
from uhf import UHF


class OS(object):

	def __init__(self,basis):

		self.basis = basis
		self.S     = np.zeros(( basis.nao(),basis.nao() ))
		self.T     = np.zeros(( basis.nao(),basis.nao() ))
		self.D     = np.zeros(( basis.nao(),basis.nao() ))

	def computeOverlap(self):

		basis  = self.basis

		icount = 0
		for ishell in range(basis.nshell()):
			aShell = basis.shell(ishell)

			for i in range(aShell.am*2+1):
				jcount = 0

				for jshell in range(basis.nshell()):
					bShell = basis.shell(jshell)

					for j in range(bShell.am*2+1):

						for aprimitive in range(aShell.nprimitive):
							for bprimitive in range(bShell.nprimitive):

								alphaA = aShell.exp(aprimitive)
								alphaB = bShell.exp(bprimitive)
								alpha  = alphaA + alphaB
								zeta   = alphaA*alphaB/(alphaA+alphaB)

								A      = np.array( [mol.x(aShell.ncenter), mol.y(aShell.ncenter) ,mol.z(aShell.ncenter)] )
								B      = np.array( [mol.x(bShell.ncenter), mol.y(bShell.ncenter) ,mol.z(bShell.ncenter)] )
								P      = (alphaA*A + alphaB*B)/(alphaA+alphaB)

								Ca     = aShell.coef(aprimitive)
								Cb     = bShell.coef(bprimitive)

								kappa  = np.exp( -zeta*np.linalg.norm(A-B)**2 )
								KK     = kappa*(np.pi/alpha)**(3/2)

								XYZ    = self.osRecursion(P-A,P-B,alpha,aShell.am,bShell.am)
								x,y,z  = XYZ[0], XYZ[1], XYZ[2]
								
								indices = [0,0,0,0,0,0]	
							
								if aShell.am > 0:
									indices[i] += 1

								if bShell.am > 0:
									indices[j+3] += 1

								self.S[icount,jcount] += KK*Ca*Cb*x[indices[0],indices[3]]*y[indices[1],indices[4]]*z[indices[2],indices[5]]

						jcount += 1
				icount += 1

		return self.S

	def osRecursion(self,PA, PB, alpha, AMa, AMb):

		xyz   = []
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


if __name__ == '__main__':
	config = configparser.ConfigParser()
	config.read('Options.ini')

	mol   = psi4.geometry(config['DEFAULT']['molecule'])
	basis = psi4.core.BasisSet.build(mol, 'BASIS', 'STO-3G', puream=0)
	mints = psi4.core.MintsHelper(basis)

	uhf   = UHF(mol,mints)
	uhf.loadIntegrals()
	
	integrals = OS(basis)
	print( integrals.computeOverlap() - uhf.S )
