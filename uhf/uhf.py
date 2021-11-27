#!/usr/bin/env python3
import psi4, numpy as np, configparser


class UHF:

	def __init__( self, options ):

		self.mol   = psi4.geometry( options['DEFAULT']['molecule'] )
		self.mol.update_geometry()

		self.basisName = options['DEFAULT']['basis']
		self.basis = psi4.core.BasisSet.build(self.mol, "BASIS", self.basisName ,puream=0)
		self.getIntegrals() 

		scf = options['SCF']
		self.conv = 10**( -int( scf['conv']) )
		self.maxiter = int( scf['max_iter'] )

		mult     = self.mol.multiplicity()
		nelec    = self.getNelec(self.mol)       
		self.Vnu = self.mol.nuclear_repulsion_energy()
		self.Na  = int( 0.5*(nelec+mult-1) )
		self.Nb  = nelec - self.Na
		self.E   = 0

		self.getIntegrals()


	def getNelec(self,mol):

		char = mol.molecular_charge()
		nelec = -char
		for A in range(mol.natom()):
			nelec += mol.Z(A)

		return int(nelec)


	def getIntegrals(self):
		
		mints  = psi4.core.MintsHelper(self.basis)

		self.V = np.array( mints.ao_potential() )
		self.T = np.array( mints.ao_kinetic() )
		self.G = np.array( mints.ao_eri() )
		self.G = self.G.transpose((0,2,1,3)) - self.G.transpose((0,2,3,1))

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

		print('           Iter         Energy                   ΔE                 ‖ΔD‖')
		print('---------------------------------------------------------------------------')
		for i in range(self.maxiter):

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

			E0  = self.E
			E   = np.trace( (H+0.5*va).dot(Da) ) + np.trace( (H+0.5*vb).dot(Db) ) + self.Vnu
			dE  = np.fabs(E-E0)
			dDa = np.fabs( np.linalg.norm(Da) - np.linalg.norm(self.Da) )
			dDb = np.fabs( np.linalg.norm(Db) - np.linalg.norm(self.Db) )
			dD  = (dDa + dDb)/2
			
			if __name__ == '__main__':
				print("@UHF-iter {:>4} {:>20.10f}{:>20.10f}{:>20.10f}".format(i,E,dE,dD))

			if dE < self.conv:
				self.converged = True
				break

			##  save
			self.Da = Da
			self.Db = Db
			self.E  = E

		return E


	def psiSCF(self):

		psi4.core.set_output_file("output.dat",False)

		psi4.set_options({'basis':  self.basisName,
						  'scf_type': 'pk',
						  'reference': 'uhf',
						  'puream': 0,
						  'print': 0 })

		return psi4.energy('scf')


if __name__ == '__main__':

	config = configparser.ConfigParser()
	config.read('Options.ini')
	uhf   = UHF(config)
	print( uhf.computeEnergy() )
	print( uhf.psiSCF() )
