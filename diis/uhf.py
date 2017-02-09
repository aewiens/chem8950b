#!/usr/bin/env python3
import psi4, numpy as np, configparser as cfp

class UHF(object):

	def __init__(self,molecule,mints,maxiter,conv,diis=1,diisStart=4,diisNvector=6):

		self.maxiter = int(maxiter)
		self.conv    = float(conv)

		self.Na   = self.getNelec(molecule,spin='alpha')
		self.Nb   = self.getNelec(molecule,spin='beta')
		self.Vnu  = molecule.nuclear_repulsion_energy()
		self.E    = 0
		self.diis = int(diis)

		self.getIntegrals(mints)

		if diis:
			self.diisStart = int(diisStart)
			self.diisNvec  = int(diisNvector)
			self.focks = []
			self.densities = []

		self.converged = False


	def getIntegrals(self,mints):
		self.V = np.array( mints.ao_potential() )
		self.T = np.array( mints.ao_kinetic() )
		self.G = np.array( mints.ao_eri() ).transpose((0,2,1,3))
		self.S = np.array( mints.ao_overlap() )

		S = mints.ao_overlap()
		S.power(-0.5, 1.e-16)
		self.X = S.to_array() 

		self.Da = np.zeros_like(self.S)
		self.Db = np.zeros_like(self.S)
	
		return None


	def computeEnergy(self):

		H  = self.T + self.V
		G  = self.G
		Da = self.Da
		Db = self.Db
		X  = self.X

		for i in range(self.maxiter):

			va = np.einsum("mnrs,ns->mr",G,Da) - np.einsum("mnsr,ns->mr",G,Da) + np.einsum("mnrs,ns->mr",G,Db)
			vb = np.einsum("mnrs,ns->mr",G,Db) - np.einsum("mnsr,ns->mr",G,Db) + np.einsum("mnrs,ns->mr",G,Da)
			fa = H + va
			fb = H + vb

			if self.diis and i >= self.diisStart:

				self.focks.append((fa,fb))
				self.densities.append((Da,Db))

				N = len(self.focks)
				start = 0
				if N > self.diisNvec:
					start = N - self.diisNvec

				focks = self.focks[start:N]
				densities = self.densities[start:N]

				fa, fb = self.solveDIIS(focks,densities)	

			tfa = X@fa@X
			tfb = X@fb@X

			ea, tCa = np.linalg.eigh(tfa)
			eb, tCb = np.linalg.eigh(tfb)

			Ca  = X@tCa
			Cb  = X@tCb
			oCa = Ca[:,:self.Na]
			oCb = Cb[:,:self.Nb]

			Da = oCa@oCa.T
			Db = oCb@oCb.T

			E0 = self.E
			E  = np.trace((fa-0.5*va)@Da) + np.trace( (fb-0.5*vb)@Db ) + self.Vnu
			dE = np.fabs(E-E0)

			if __name__ == '__main__':
				print("UHF  {:>4} {: >21.11}  {: >21.11}".format(i,E,dE))

			if dE < self.conv:
				self.converged = True

				if __name__ == '__main__':
					print("\nUHF procedure has converged.\nFinal UHF energy:{:20.11f}".format(self.E) )
				break

			self.Da = Da
			self.Db = Db
			self.E  = E

		return E



	def getNelec(self,mol,spin='alpha'):

		mult   = mol.multiplicity()
		charge = mol.molecular_charge()
		nelec  = -charge

		for A in range(mol.natom()):
			nelec += mol.Z(A)

		Na =  0.5*(nelec+mult-1)

		if   spin == 'alpha':   
			return int(Na)

		elif spin == 'beta':
			return int(nelec - Na)



	def solveDIIS(self,focks,densities):

		S  = self.S
		errorVectors = []
		for (fa,fb), (da,db) in zip(focks,densities):
			ea = fa@da@S - S@da@fa
			eb = fb@db@S - S@db@fb
			errorVectors.append( (ea,eb) )

		N  = len(errorVectors)
		P  = np.zeros((N+1,N+1))

		P[:-1,-1] = -1
		P[-1,:-1] = -1
	
		for i, (iEVA,iEVB)  in enumerate(errorVectors):
			for j, (jEVA,jEVB) in enumerate(errorVectors):
				P[i,j]  = np.vdot(iEVA,jEVA) + np.vdot(iEVB, jEVB)
				P[i,j] /= 2

		f = np.array( [0]*N + [-1] )
		q = np.linalg.solve(P, f)
		
		weightFocks = [ (fa*q, fb*q) for q,(fa,fb) in zip(q,focks) ]

		return tuple([sum(i) for i in zip(*weightFocks)])



if __name__ == '__main__':

	config = cfp.ConfigParser()
	config.read('Options.ini')

	molecule = psi4.geometry( config['DEFAULT']['molecule'] )
	molecule.update_geometry()

	basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'],puream=0)
	mints = psi4.core.MintsHelper(basis)
	scf   = config['SCF']

	uhf   = UHF(molecule,
			  mints,
			  scf['maxIter'],
			  scf['conv'],
			  scf['diis'],
			  scf['diis_start'],
			  scf['diis_nvector'])

	uhf.computeEnergy()
