#!/usr/bin/env python3
import psi4, numpy as np, configparser


class RHF:

	def __init__(self,molecule,mints,maxiter,conv,diis=False,diisStart=0,diisNvector=0):

		self.docc = self.getDocc(molecule)       
		self.Vnu  = molecule.nuclear_repulsion_energy()
		self.E    = 0

		self.getIntegrals(mints)
		self.maxiter = maxiter
		self.conv    = conv
		self.diis    = diis

		if diis:
			self.diisStart = diisStart
			self.Ndiis  = diisNvector

		self.converged = False


	def getIntegrals(self,mints):

		self.V = np.array( mints.ao_potential() )
		self.T = np.array( mints.ao_kinetic() )
		self.G = np.array( mints.ao_eri() ).transpose((0,2,1,3))

		S = mints.ao_overlap()
		S.power(-0.5, 1.e-16)
		self.X = S.to_array() 
		self.S = np.array( mints.ao_overlap() )
		self.D = np.zeros_like( self.S )


	def computeEnergy(self):

		H  = self.T + self.V
		G  = self.G
		X  = self.X
		S  = self.S
		D  = self.D
		docc = self.docc

		if self.diis:
			focks = []
			densities = []

		for i in range(self.maxiter):

			J = np.einsum("ikjl,kl",G,D)
			K = np.einsum("iklj,kl",G,D)
			F = H+2*J-K

			if self.diis and i >= self.diisStart:
				focks.append(F)
				densities.append(D)
				F = self.solveDIIS(S,focks,densities)

			e, tC = np.linalg.eigh( X@F@X )
			C     = X@tC
			Cocc  = C[:,:docc]
			D     = Cocc@Cocc.T

			E0 = self.E
			E  = np.trace((H+F) @ D)+ self.Vnu
			dE = np.fabs(E-E0)

			if __name__ == '__main__':
				print("RHF  {:>4} {: >21.11}  {: >21.11}".format(i,E,dE))

			if dE < self.conv:
				self.converged = True

				if __name__ == '__main__':
					self.writeOutput()
				break

			self.D = D
			self.E = E
			self.C = C 
			self.e = e

		return E

	def writeOutput(self):

		if self.converged:
			print("\nRHF procedure has converged.\nFinal RHF energy:{:20.11f}".format(self.E) )
		if not self.converged:
			print("RHF procedure did not converge. Sorry")


	def getDocc(self,mol):
		char = mol.molecular_charge()
		nelec = -char
		for A in range(mol.natom()):
			nelec += mol.Z(A)
		return int(nelec//2)


	def solveDIIS(self,S,focks,densities):

		if len(focks) > self.Ndiis:
			start     = len(focks)-self.Ndiis
			densities = densities[start:len(focks)]
			focks     = focks[start:len(focks)]

		errorVectors = [ F@D@S - S@D@F for F,D in zip(focks,densities) ]

		N = len(errorVectors)
		P = np.zeros(( N+1, N+1))

		for i in range(N):
			P[-1,i] = P[i,-1] = -1
			for j in range(N):
				P[i,j] = np.vdot(errorVectors[i],errorVectors[j])

		f = np.array( [0]*N + [-1] )
		q = np.linalg.solve(P,f)

		return sum( [q[i]*F for i,F in enumerate(focks)] )


if __name__ == '__main__':

	config = configparser.ConfigParser()
	config.read('Options.ini')
	molecule   = psi4.geometry( config['DEFAULT']['molecule'] )
	molecule.update_geometry()


	basis     = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'])
	mints     = psi4.core.MintsHelper(basis)

	scfConfig = config['SCF']
	maxiter   = int( scfConfig['maxIter'] )
	scfConv   = float( scfConfig['conv']  )
	diis      = bool( scfConfig['diis'] )
	diisNvec  = int( scfConfig['diis_nvector'] ) 
	diisStart = int( scfConfig['diis_start'] )

	rhf   = RHF(molecule,
				mints,
				maxiter,
				scfConv,
				diis=diis,
				diisStart=diisStart,
				diisNvector=diisNvec)

	rhf.computeEnergy()
