#!/usr/bin/env python3
import psi4, numpy as np, configparser as cfp


class UHF:

	def __init__(self,options):

		mol = psi4.geometry( options['DEFAULT']['molecule'] )
		mol.update_geometry()

		self.basisName  = options['DEFAULT']['basis']
		self.basis = psi4.core.BasisSet.build(mol, "BASIS", self.basisName, puream=0)
		self.mints = psi4.core.MintsHelper(self.basis)
		self.getIntegrals()

		self.conv    = 10**( -int( options['SCF']['conv'] ))
		self.dConv   = 10**( -int( options['SCF']['d_conv'] ))
		self.maxiter = int( options['SCF']['max_iter'] )
		self.norb    = len(self.S)
		self.nelec   = self.getNelec(mol)
		self.nvirt   = self.norb - self.nelec

		self.Vnu = mol.nuclear_repulsion_energy()
		self.D   = np.zeros_like(self.S)
		self.E   = 0.0
		self.mol = mol


	def getIntegrals(self):
		mints  = self.mints
		self.T = self.block_oei( mints.ao_kinetic() )
		self.V = self.block_oei( mints.ao_potential() )
		self.S = self.block_oei( mints.ao_overlap() )                      

		S = mints.ao_overlap()
		S.power(-0.5,1.e-16)
		self.X = self.block_oei( S )

		G = self.block_tei(np.array( mints.ao_eri() ) )
		self.G = G.transpose((0,2,1,3))-G.transpose((0,2,3,1))


		
	def computeEnergy(self):

		H = self.T + self.V
		X = self.X
		G = self.G
		D = self.D
		F = np.zeros_like(H)

		for i in range(self.maxiter):

			v = np.einsum("mnrs,ns->mr", G, self.D)
			F = H + v
			e,tC = np.linalg.eigh(X@F@ X)

			C  = X@tC
			oC = C[:,:self.nelec]
			D  = oC@oC.T

			E0 = self.E
			E  = np.trace( (H+0.5*v)@D) + self.Vnu
			dE = np.fabs(E-E0)

			dD = np.fabs( np.linalg.norm(D) - np.linalg.norm(self.D) )
			
			if __name__ == '__main__':
				print("@UHF-iter {:>4}{: >20.12f}{: >20.12f}{: >20.12f}".format(i,E,dE,dD))

			self.E = E
			self.C = C
			self.e = e
			self.D = D 
			self.F = F

			if dE < self.conv and dD < self.dConv:
				break

		return self.E


	def getNelec(self, mol):

		char = mol.molecular_charge()
		nelec = -char
		for A in range(mol.natom()):
			nelec += mol.Z(A)

		return int(nelec)


	"""
	spin-blocking functions: transform from spatial orbital {x_mu} basis to spin orbital basis {x_mu alpha, x_mu beta}
	"""
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


if __name__ == '__main__':
	
	config = cfp.ConfigParser()
	config.read('Options.ini')
	uhf = UHF(config)
	uhf.computeEnergy()
