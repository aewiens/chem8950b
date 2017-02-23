#!/usr/bin/env python3
import psi4, sys, numpy as np, configparser
from uhf import UHF


class MP2:

	def __init__(self,options):

		psi4.core.set_output_file("output.dat",False)

		self.Ec    = 0.0
		uhf = UHF(options)
		self.E0    = uhf.computeEnergy()
		self.nocc  = uhf.nelec
		self.norb  = uhf.norb
		self.e     = uhf.e
		self.C     = uhf.C
		self.G     = uhf.G


	def tei_einsum(self,G,C):
		g   = np.einsum("Ss,PQRS->PQRs",self.C,self.G)
		gp  = np.einsum("Rr,PQRs->PQrs",self.C,g)
		gpp = np.einsum("Qq,PQrs->Pqrs",self.C,gp)
		GMO = np.einsum("Pp,Pqrs->pqrs",self.C,gpp)
		return GMO


	def computeEnergy(self):
		"""
		Spin-orbital implementation of mp2 equations
		"""

		Gmo  = self.tei_einsum( self.G, self.C )
		e    = self.e
		Ec   = 0.0

		for i in range( self.nocc ):
			for j in range( self.nocc ):
				for a in range( self.nocc,self.norb ):
					for b in range( self.nocc,self.norb ):
						Ec += 0.25*(Gmo[i,j,a,b]*Gmo[a,b,i,j])/(e[i]+e[j]-e[a]-e[b])

		return self.E0 + Ec


	def psiMP2(self,options):
		psi4.set_options({'basis': options['DEFAULT']['basis'],
						  'scf_type': 'pk',
						  'mp2_type': 'conv',
						  'reference': 'uhf',
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
	print(mp2.computeEnergy() )
