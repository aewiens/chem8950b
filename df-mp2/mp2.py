#!/usr/bin/env python3 
import psi4, sys, numpy as np, configparser
from uhf import UHF

class MP2:

	def __init__(self,options):

		uhf = UHF(options)

		self.E0    = uhf.computeEnergy()
		self.e     = uhf.e
		self.C     = uhf.C
		self.nocc  = uhf.nelec
		self.norb  = uhf.norb
		self.uhf   = uhf

		df = int(options['MP2']['df'])
		
		if df:
			dfBasisName  = options['DEFAULT']['df_basis']
			dfBasis = psi4.core.BasisSet.build( uhf.mol, "DF_BASIS_MP2", dfBasisName, puream=0)
			self.G  = self.densityFit(options,dfBasis)

		else:
			self.G = self.transformTEI( uhf.G, uhf.C)


	def transformTEI(self,G,C):
		g   = np.einsum("Ss,PQRS->PQRs",C,G)
		gp  = np.einsum("Rr,PQRs->PQrs",C,g)
		gpp = np.einsum("Qq,PQrs->Pqrs",C,gp)
		GMO = np.einsum("Pp,Pqrs->pqrs",C,gpp)
		return GMO


	def densityFit(self,options,dfBasis):
		uhf   = self.uhf
		basis = uhf.basis
		mints = uhf.mints 
		zero  = psi4.core.BasisSet.zero_ao_basis_set()
	
		J = mints.ao_eri( dfBasis,zero,dfBasis,zero )
		J.power( -0.5, 1.e-16 )
		J = np.squeeze(J)

		B   = np.squeeze( mints.ao_eri( dfBasis,zero,basis,basis ))
		b   = [ self.uhf.block_oei(P) for P in B ]	
		bAO = np.einsum("PQ,Ppq->Qpq",J, np.array(b) )
		bAO = np.einsum("nq,Qmn->Qmq",self.C,bAO)
		bMO = np.einsum("mp,Qmq->Qpq",self.C,bAO)
		GMO = np.einsum("Qpr,Qqs->pqrs",bMO,bMO)
		
		return GMO - GMO.transpose((0,1,3,2))
		

	def computeEnergy(self):
		e    = self.e
		Ec   = 0.0

		for i in range( self.nocc ):
			for j in range( self.nocc ):
				for a in range( self.nocc,self.norb ):
					for b in range( self.nocc,self.norb ):
						Ec += 0.25 * (self.G[i,j,a,b] * self.G[a,b,i,j]) / (e[i]+e[j]-e[a]-e[b])

		return self.E0 + Ec


	def psiMP2(self,options):
		psi4.set_options({'basis': options['DEFAULT']['basis'],
						  'scf_type': 'pk',
						  'reference': 'uhf',
						  'df_basis_mp2' : 'cc-pvdz-ri',
						  'puream': 0,
						  'print': 0 })

		return psi4.energy('mp2')


if __name__ == '__main__':
	
	config = configparser.ConfigParser()
	config.read('Options.ini')
	mp2 = MP2( config )
	print( mp2.psiMP2(config) )
	print( mp2.computeEnergy() )
