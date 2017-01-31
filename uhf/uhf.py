#!/usr/bin/env python3

import psi4, numpy as np, configparser
from  scipy import linalg as la


class UHF(object):

	def __init__(self,config):

		config.read('Options.ini')
		self.maxit = int(config['SCF']['maxIter'])
		molecule   = psi4.geometry( config['DEFAULT']['molecule'] )

		molecule.update_geometry()

		mult     = molecule.multiplicity()
		nelec    = self.get_nelec(molecule)       
		self.Vnu = molecule.nuclear_repulsion_energy()
		self.Na  = int( 0.5*(nelec+mult-1) )
		self.Nb  = nelec - self.Na
		self.E   = 0
		self.mol = molecule


	def get_nelec(self,mol):
		char = mol.molecular_charge()
		nelec = -char
		for A in range(mol.natom()):
			nelec += mol.Z(A)
		return int(nelec)


	def loadIntegrals(self):

		basis   = psi4.core.BasisSet.build(self.mol, "BASIS", config['DEFAULT']['basis'])
		mints   = psi4.core.MintsHelper(basis)

		V = np.array( mints.ao_potential() )
		T = np.array( mints.ao_kinetic() )
		G = np.array( mints.ao_eri() )

		##  class variables for UHF iterations
		self.S     = np.array( mints.ao_overlap() )
		self.Hcore = T + V
		self.G     = G.transpose((0,2,1,3))
		self.X     = np.matrix( la.funm(self.S, lambda x : x**(-0.5) ) )
		self.Da    = np.zeros(self.X.shape)
		self.Db    = np.zeros(self.X.shape)

		return None


	def compute_energy(self):

		self.loadIntegrals()

		H  = self.Hcore
		G  = self.G
		Da = self.Da
		Db = self.Db
		X  = self.X

		self.converged = False
		for i in range( self.maxit ):

			va = np.einsum("mnrs,ns->mr",G,Da) - np.einsum("mnsr,ns->mr",G,Da) + np.einsum("mnrs,ns->mr",G,Db)
			vb = np.einsum("mnrs,ns->mr",G,Db) - np.einsum("mnsr,ns->mr",G,Db) + np.einsum("mnrs,ns->mr",G,Da)

			Fa = H + va
			Fb = H + vb

			tFa = X.dot(Fa.dot(X))
			tFb = X.dot(Fb.dot(X))

			ea, tCa = la.eigh(tFa)
			eb, tCb = la.eigh(tFb)

			Ca = X.dot(tCa)
			Cb = X.dot(tCb)

			oCa = Ca[:,:self.Na]
			oCb = Cb[:,:self.Nb]

			Da = oCa.dot(oCa.T)
			Db = oCb.dot(oCb.T)

			E0 = self.E
			E = np.trace( (H+0.5*va).dot(Da) ) + np.trace( (H+0.5*vb).dot(Db) ) + self.Vnu
			dE = np.fabs(E-E0)

			print("UHF  {:>4} {: >21.11}  {: >21.11}".format(i,E,dE))

			if dE < 1e-12:
				self.converged = True
				break

			##  save
			self.Da = Da
			self.Db = Db
			self.E  = E

		self.write_output()
		return E


	def write_output(self):
		if self.converged:
			print("\nUHF procedure has converged.\nFinal UHF energy:{:20.11f}".format(self.E) )

		if not self.converged:
			print("UHF procedure did not converge. Sorry")

		return None


if __name__ == '__main__':

	config = configparser.ConfigParser()
	uhf    = UHF(config)
	uhf.compute_energy()
