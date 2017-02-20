#!/usr/bin/env python3
import psi4, sys, numpy as np, configparser
from uhf import UHF

class MP2:

	def __init__(self,options):

		uhf = UHF(options)
		self.nocc  = uhf.nelec
		self.norb  = uhf.norb
		self.Ec    = 0.0
		self.E0    = uhf.computeEnergy()
		self.e     = uhf.e
		self.C     = uhf.C
		self.G     = uhf.G


	def tei_einsum(self,g,C):
		return np.einsum("Pp,Pqrs->pqrs",C,
				np.einsum("Qq,PQrs->Pqrs",C,
				np.einsum("Rr,PQRs->PQrs",C,
				np.einsum("Ss,PQRS->PQRs",C,g))))


	def computeEnergy(self):
		"""
		Spin-orbital implementation of mp2 equations
		"""

		Gmo  = self.tei_einsum(self.G, self.C)
		e    = self.e
		Ec   = 0.0

		for i in range( self.nocc ):
			for j in range( self.nocc ):
				for a in range( self.nocc,self.norb ):
					for b in range( self.nocc,self.norb ):
						Ec += 0.25*(Gmo[i,j,a,b]*Gmo[a,b,i,j])/(e[i]+e[j]-e[a]-e[b])

		return self.E0 + Ec


	def psiMP2(self):
		return None


if __name__ == '__main__':
	
	config = configparser.ConfigParser()
	config.read('Options.ini')
	mp2 = MP2( config )
	print(mp2.computeEnergy() )
