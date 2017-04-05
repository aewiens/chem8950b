#!/usr/bin/env python3
import psi4, sys, numpy as np, configparser
from uhf import UHF


class MP2:

	def __init__(self,options):

		self.options = options

		self.Ec    = 0.0
		uhf = UHF(options)
		self.E0    = uhf.computeEnergy()
		self.nocc  = uhf.nelec
		self.norb  = uhf.norb
		self.mol   = uhf.mol
		self.e     = uhf.e
		self.C     = uhf.C
		self.uhf   = uhf
		self.df    = int( options['MP2']['df'] )
		self.GMO   = self.transformTEI( uhf.G, uhf.C )	


	def transformTEI(self,G,C):

		if not self.df:
			g   = np.einsum("Ss,PQRS->PQRs",C,G)
			gp  = np.einsum("Rr,PQRs->PQrs",C,g)
			gpp = np.einsum("Qq,PQrs->Pqrs",C,gp)
			GMO = np.einsum("Pp,Pqrs->pqrs",C,gpp)
			return GMO

		elif self.df:
			dfBasisName = self.options['MP2']['df_basis']
			dfBasis     = psi4.core.BasisSet.build( self.mol, "DF_BASIS_MP2", dfBasisName, puream=0 )
			return self.densityFit( self.options, dfBasis )


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
		"""
		Spin-orbital implementation of mp2 equations
		"""
		Gmo  = self.GMO
		e    = self.e
		self.Ec   = 0.0

		for i in range( self.nocc ):
			for j in range( self.nocc ):
				for a in range( self.nocc,self.norb ):
					for b in range( self.nocc,self.norb ):
						self.Ec += 0.25*(Gmo[i,j,a,b]*Gmo[a,b,i,j])/(e[i]+e[j]-e[a]-e[b])

		return self.E0 + self.Ec


	def psiMP2(self,options):

		psi4.core.set_output_file("output.dat",False)

		psi4.set_options({'basis': options['DEFAULT']['basis'],
						  'scf_type': 'pk',
						  'reference': 'uhf',
						  'df_basis_mp2' : options['MP2']['df_basis'],
						  'puream': 0,
						  'print': 0 })

		return psi4.energy('mp2')


def block_tei(T):
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
	
	config = configparser.ConfigParser()
	config.read('Options.ini')
	mp2 = MP2( config )
	print( mp2.psiMP2(config) )
	print( mp2.computeEnergy() )
	#print( "{:.12f}".format( mp2.computeEnergy() ) )
