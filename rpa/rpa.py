#!/usr/bin/env python3
import sys, configparser, numpy as np
from scipy import linalg as la
sys.path.insert(0,"../mp2/")
from uhf import UHF
from mp2 import MP2


class CIS:

	def __init__(self,options):

		uhf = UHF( options )

		uhf.computeEnergy()

		self.nocc  = uhf.nocc
		self.nvirt = uhf.norb - self.nocc
		self.E0    = uhf.E
		self.e     = uhf.e

		mp2 = MP2( options )
		self.GMO   = mp2.transformTEI( uhf.G, uhf.C )

		print("")
		print("--------------------------------" )
		print( "| Output for TDHF/RPA Procedure |" )
		print("--------------------------------" )
		print( "\n @UHF Energy:   %f\n\n" % uhf.E )


	def getSingles(self):
		"""  return a list of all (i,a) single excitations  xi_i -> xi_a
		"""
		return [(i,a) for i in range(self.nocc) for a in range(self.nocc,self.nocc+self.nvirt)]


	def computeStates(self):

		E0 = self.E0
		e  = self.e
		nDeterminants = self.nocc*self.nvirt
		excitations   = self.getSingles()

		##  build RPA hamiltonian blocks  ##
		A = np.zeros((nDeterminants,nDeterminants))
		for P, (i,a) in enumerate(excitations):
			for Q, (j,b) in enumerate(excitations):
				A[P,Q] = self.GMO[a,j,i,b] + (e[a] - e[i])*(a==b)*(i==j) 

		B = np.zeros((nDeterminants,nDeterminants))
		for P, (i,a) in enumerate(excitations):
			for Q, (j,b) in enumerate(excitations):
				B[P,Q] = self.GMO[a,b,i,j]

		fullH    = np.bmat( [[A,B],[-B,-A]] )
		reducedH = (A+B)@(A-B)

		fE  = sorted( np.linalg.eigvals(fullH) )
		fE = fE[nDeterminants:]
		rEV = np.linalg.eigvals(reducedH)

		rE   = sorted( [ np.sqrt(ev) for ev in rEV ] )

		print("-------------------------------------------------")
		print( "| RPA Energy Levels // Full-Dimensional Approach |" )
		print("-------------------------------------------------")
		print("   State    Energy (Eh)   ")
		print("---------------------------")
		for i, energy in enumerate(fE):
			print("{:4d}  {: >16.11f}  ".format(i+1,energy) )
		print("---------------------------")

		print("---------------------------------------------------")
		print( "| RPA Energy Levels // Reduced-Dimension Approach |" )
		print("---------------------------------------------------")
		print("   State    Energy (Eh)   ")
		print("---------------------------")
		for i, energy in enumerate(rE):
			print("{:4d}  {: >16.11f}  ".format(i+1,energy) )
		print("---------------------------")

if __name__ == '__main__':
	
	config = configparser.ConfigParser()
	config.read('Options.ini')

	cis = CIS(config)
	cis.computeStates()

