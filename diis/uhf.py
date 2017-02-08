#!/usr/bin/env python3
import psi4, numpy as np, configparser as cfp


class UHF(object):

	def __init__(self,molecule,mints,maxiter,conv,diis=False,diisStart=0,diisNvector=0):

		self.maxiter = maxiter 
		self.conv    = conv

		self.Na   = self.getNelec(molecule,spin='alpha')
		self.Nb   = self.getNelec(molecule,spin='beta')
		self.Vnu  = molecule.nuclear_repulsion_energy()
		self.E    = 0
		self.diis = diis

		self.getIntegrals(mints)

		if diis:
			self.diisStart = diisStart
			self.diisNvec  = diisNvector
			self.Falphas   = []
			self.Fbetas    = []
			self.Dalphas   = []
			self.Dbetas    = []

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

				self.Falphas.append(fa)
				self.Dalphas.append(Da)
				self.Fbetas.append(fb)
				self.Dbetas.append(Db)

				diisInfo = self.diisHelper( [self.Falphas, self.Dalphas, self.Fbetas, self.Dbetas ] )
				fa, fb   = self.solveDIIS(*diisInfo)
			
			tfa  = X@fa@X
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
					self.writeOutput()
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

		if   spin== 'alpha':   return int(Na)
		elif spin== 'beta':    return int(nelec - Na)


	def diisHelper(self,ListOfArrays):

		N = len(ListOfArrays[0])
		if N  <= self.diisNvec:
			return ListOfArrays

		elif N > self.diisNvec:
			start = N - self.diisNvec
			return [ array[start:N] for array in ListOfArrays ]


	def diisP(self,focks,densities):

		S = self.S
		errorVectors = [ F@D@S - S@D@F for F,D in zip(focks,densities) ]
		N = len(errorVectors)
		P = np.zeros(( N+1, N+1))

		for j in range(N):
			P[-1,j] = P[j,-1] = -1
			for k in range(N):
				P[j,k] = np.vdot(errorVectors[j],errorVectors[k])

		return P


	def solveDIIS(self,Falphas,Dalphas,Fbetas,Dbetas):
		
		Pa = self.diisP( Falphas, Dalphas )
		Pb = self.diisP( Fbetas, Dbetas )

		f = np.zeros(( Pa.shape[0] ))
		f[-1] = -1

		q = np.linalg.solve( (Pa+Pb)/2, f)

		fa = sum( [q[i]*F for i,F in enumerate(Falphas)] )
		fb = sum( [q[i]*F for i,F in enumerate(Fbetas)] )

		return fa, fb


	def writeOutput(self):

		if self.converged:
			print("\nUHF procedure has converged.\nFinal UHF energy:{:20.11f}".format(self.E) )
		if not self.converged:
			print("UHF procedure did not converge. Sorry")


if __name__ == '__main__':

	config = cfp.ConfigParser()
	config.read('Options.ini')

	molecule = psi4.geometry( config['DEFAULT']['molecule'] )
	molecule.update_geometry()

	basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'],puream=0)
	mints = psi4.core.MintsHelper(basis)

	diis      = bool( config['SCF']['diis'] )
	diisNvec  = int( config['SCF']['diis_nvector'] ) 
	diisStart = int( config['SCF']['diis_start'] )
	maxiter   = int( config['SCF']['maxIter'] )
	conv      = float( config['SCF']['conv']  )

	uhf = UHF(molecule,mints,maxiter,conv,diis=diis,diisStart=diisStart,diisNvector=diisNvec)
	uhf.computeEnergy()
