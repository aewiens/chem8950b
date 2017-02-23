#!/usr/bin/env python3
import psi4, numpy as np, configparser


class UHF:

	def __init__(self,options):

		mol = psi4.geometry( options['DEFAULT']['molecule'] )
		mol.update_geometry()

		basisName  = options['DEFAULT']['basis']
		self.basis = psi4.core.BasisSet.build(mol,"BASIS",basisName,puream=0)
		self.getIntegrals( self.basis )

		self.norb    = len(self.S)
		self.nelec   = self.getNelec(mol)

		scf = options['SCF']
		self.conv    = 10**( -int( scf['conv'] ) )
		self.maxiter = int( scf['max_iter'] )
		self.diis    = int( scf['diis'] )


		self.Vnu = mol.nuclear_repulsion_energy()
		self.E   = 0.0
		self.D   = np.zeros_like(self.S)
		self.mol = mol

		if self.diis:
			self.diisStart = int( scf['diis_start'])
			self.Ndiis     = int( scf['diis_nvector'] )


	def getIntegrals(self,basis):

		mints = psi4.core.MintsHelper(basis)

		##  overlap  ##
		self.S = self.block_oei( mints.ao_overlap() )                      
		S = mints.ao_overlap()
		S.power(-0.5,1.e-16)
		self.X = self.block_oei( S )

		##   one-electron  ##
		self.T = self.block_oei( mints.ao_kinetic() )
		self.V = self.block_oei( mints.ao_potential() )

		##  2-electron (physicist notation)  ##
		G = self.block_tei(np.array( mints.ao_eri() ) )
		self.G = G.transpose((0,2,1,3)) - G.transpose((0,2,3,1))

		self.mints = mints
	
	
	def computeEnergy(self):

		H = self.T + self.V
		X = self.X
		G = self.G
		D = self.D

		if self.diis:
			focks = []
			densities = []

		for i in range(self.maxiter):

			v  = np.einsum("mnrs,ns->mr",G,self.D)
			F  = H + v

			if self.diis and i >= self.diisStart:
				focks.append(F)
				densities.append(D)
				F = self.solveDIIS( self.S,focks,densities )

			e,tC = np.linalg.eigh(X@F@ X)

			C  = X@tC
			oC = C[:,:self.nelec]
			D  = oC@oC.T

			E0 = self.E
			E  = np.trace( (H+0.5*v)@D) + self.Vnu
			dE = np.fabs(E-E0)
			
			if __name__ == '__main__':
				print("UHF  {:>4} {: >21.13}  {: >21.13}".format(i,E,dE))

			self.E = E
			self.C = C
			self.e = e
			self.D = D 

			if dE < self.conv:
				break

		return self.E


	def getNelec(self, mol):

		char = mol.molecular_charge()
		nelec = -char
		for A in range(mol.natom()):
			nelec += mol.Z(A)

		return int(nelec)


	def block_oei(self,A):
		A = np.matrix(A)
		O = np.zeros(A.shape)
		return np.bmat( [[A,O],[O,A]] )


	def block_tei(self,T):
		t = np.array(T)
		n = t.shape[0]
		I2 = np.identity(2)
		T = np.zeros( (2*n,2*n,2*n,2*n) )
		for p in range(n):
			for q in range(n):
				T[p,q] = np.kron( I2, t[p,q] )
				T[n:,n:] = T[:n,:n]
		return T


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


	def psiSCF(self,options):
		psi4.set_options({'basis': options['DEFAULT']['basis'],
						  'scf_type': 'pk',
						  'reference': 'uhf',
						  'puream': 0,
						  'print': 0 })

		return psi4.energy('scf')
			

if __name__ == '__main__':
	
	config = configparser.ConfigParser()
	config.read('Options.ini')

	uhf = UHF(config)
	print( uhf.psiSCF(config) )
	print( uhf.computeEnergy() )
