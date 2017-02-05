#!/usr/bin/env python3

import psi4, numpy as np, configparser

class UHF(object):

	def __init__(self,molecule,mints):

		mult     = molecule.multiplicity()
		nelec    = self.getNelec(molecule)       
		self.Vnu = molecule.nuclear_repulsion_energy()
		self.Na  = int( 0.5*(nelec+mult-1) )
		self.Nb  = nelec - self.Na
		self.E   = 0

		self.getIntegrals(mints)


	def getNelec(self,mol):
		char = mol.molecular_charge()
		nelec = -char
		for A in range(mol.natom()):
			nelec += mol.Z(A)
		return int(nelec)


	def getIntegrals(self,mints):

		self.V = np.array( mints.ao_potential() )
		self.T = np.array( mints.ao_kinetic() )
		self.G = np.array( mints.ao_eri() )
		self.G = self.G.transpose((0,2,1,3))

		self.S = mints.ao_overlap()
		self.S.power(-0.5, 1.e-16)
		self.X = self.S.to_array() 

		self.Da = np.zeros_like(self.S)
		self.Db = np.zeros_like(self.S)
	
		return None


	def computeEnergy(self):

		H  = self.T + self.V
		G  = self.G
		Da = self.Da
		Db = self.Db
		X  = self.X

		self.converged = False
		for i in range(100):

			va = np.einsum("mnrs,ns->mr",G,Da) - np.einsum("mnsr,ns->mr",G,Da) + np.einsum("mnrs,ns->mr",G,Db)
			vb = np.einsum("mnrs,ns->mr",G,Db) - np.einsum("mnsr,ns->mr",G,Db) + np.einsum("mnrs,ns->mr",G,Da)

			Fa = H + va
			Fb = H + vb

			tFa = X.dot(Fa.dot(X))
			tFb = X.dot(Fb.dot(X))

			ea, tCa = np.linalg.eigh(tFa)
			eb, tCb = np.linalg.eigh(tFb)

			Ca = X.dot(tCa)
			Cb = X.dot(tCb)

			oCa = Ca[:,:self.Na]
			oCb = Cb[:,:self.Nb]

			Da = oCa.dot(oCa.T)
			Db = oCb.dot(oCb.T)

			E0 = self.E
			E = np.trace( (H+0.5*va).dot(Da) ) + np.trace( (H+0.5*vb).dot(Db) ) + self.Vnu
			dE = np.fabs(E-E0)

			if __name__ == '__main__':
				print("UHF  {:>4} {: >21.11}  {: >21.11}".format(i,E,dE))

			if dE < 1e-12:

				self.converged = True
				if __name__ == '__main__':
					self.writeOutput()

				break

			##  save
			self.Da = Da
			self.Db = Db
			self.E  = E

		return E


	def writeOutput(self):

		if self.converged:
			print("\nUHF procedure has converged.\nFinal UHF energy:{:20.11f}".format(self.E) )
		if not self.converged:
			print("UHF procedure did not converge. Sorry")


if __name__ == '__main__':

	config = configparser.ConfigParser()
	config.read('Options.ini')

	molecule   = psi4.geometry( config['DEFAULT']['molecule'] )
	molecule.update_geometry()

	basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'],puream=0)
	mints = psi4.core.MintsHelper(basis)

	uhf   = UHF(molecule,mints)
	uhf.computeEnergy()
