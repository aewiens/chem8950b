#!/usr/bin/env python

import sys, configparser, psi4, numpy as np
from index import amIndices

sys.path.append('../uhf/')
from uhf import UHF

class OS(object):

	def __init__(self,basis,molecule,uhfObject):

		self.basis = basis
		self.mol   = molecule
		self.uhf   = uhfObject
		self.S     = np.zeros(( basis.nao(),basis.nao() ))
		self.T     = np.zeros(( self.S.shape ))
		self.dx    = np.zeros(( self.S.shape ))
		self.dy    = np.zeros(( self.S.shape ))
		self.dz    = np.zeros(( self.S.shape ))

		print('The number of shells: %d' % (basis.nshell()))
		print('The number of basis functions: %d' % (basis.nao()))

		for ishell in range(basis.nshell()):
			shell = basis.shell(ishell)
			print('Shell %d:' % (ishell))
			print('  Atom: %d' % (shell.ncenter))
			print('  AM: %d' % (shell.am))
			print('  # Cartesian functions: %d' % (shell.ncartesian))
			print('  # of primitive Gaussians: %d ' % (shell.nprimitive))
			print('  function_index: %d' % (shell.function_index))
			print('  center: %f, %f, %f' % (molecule.x(shell.ncenter), molecule.y(shell.ncenter), molecule.z(shell.ncenter)))


	def fillIndices(self,AM):
		
		indexObject = amIndices(AM)
		indexArray  = indexObject.indices
		indexList   = []
		for row in indexArray:
			indexList.append( [ int(i) for i in row ] )
		return indexList


	def computeKinetic(self):

		icount = 0
		for ishell in range(self.basis.nshell()):
			aShell = self.basis.shell(ishell)
			nbfA   = (aShell.am+1)*(aShell.am+2)//2

			for i in range(nbfA):
				jcount = 0

				for jshell in range(self.basis.nshell()):
					bShell = self.basis.shell(jshell)
					nbfB   = (bShell.am+1)*(bShell.am+2)//2

					for j in range(nbfB):

						for aprimitive in range(aShell.nprimitive):
							for bprimitive in range(bShell.nprimitive):

								alphaA = aShell.exp(aprimitive)
								alphaB = bShell.exp(bprimitive)
								alpha  = alphaA + alphaB
								zeta   = alphaA*alphaB/(alphaA+alphaB)

								A      = np.array( [mol.x(aShell.ncenter), mol.y(aShell.ncenter) ,mol.z(aShell.ncenter)] )
								B      = np.array( [mol.x(bShell.ncenter), mol.y(bShell.ncenter) ,mol.z(bShell.ncenter)] )
								P      = (alphaA*A + alphaB*B)/(alphaA+alphaB)

								Ca     = aShell.coef(aprimitive)
								Cb     = bShell.coef(bprimitive)

								kappa  = np.exp( -zeta*np.linalg.norm(A-B)**2 )
								KK     = kappa*(np.pi/alpha)**(3/2)

								XYZ    = self.osRecursion(P-A,P-B,alpha,aShell.am+1,bShell.am+1)
								x,y,z  = XYZ[0], XYZ[1], XYZ[2]

								aIndices = self.fillIndices(aShell.am)[i]
								bIndices = self.fillIndices(bShell.am)[j]
								l1, l2   = aIndices[0], bIndices[0]
								m1, m2   = aIndices[1], bIndices[1]
								n1, n2   = aIndices[2], bIndices[2]

								Tx = 0.5*y[m1,m2]*z[n1,n2]
								Ty = 0.5*x[l1,l2]*z[n1,n2]
								Tz = 0.5*x[l1,l2]*y[m1,m2]
							
								Tx *= l1*l2*x[l1-1,l2-1] + 4*alphaA*alphaB*x[l1+1,l2+1] - 2*alphaA*l2*x[l1+1,l2-1] - 2*alphaB*l1*x[l1-1,l2+1]
								Ty *= m1*m2*y[m1-1,m2-1] + 4*alphaA*alphaB*y[m1+1,m2+1] - 2*alphaA*m2*y[m1+1,m2-1] - 2*alphaB*m1*y[m1-1,m2+1]
								Tz *= n1*n2*z[n1-1,n2-1] + 4*alphaA*alphaB*z[n1+1,n2+1] - 2*alphaA*n2*z[n1+1,n2-1] - 2*alphaB*n1*z[n1-1,n2+1]

								self.T[icount,jcount] += KK*Ca*Cb*(Tx+Ty+Tz)

						jcount += 1
				icount += 1

		return self.T


	def computeOverlap(self):

		basis  = self.basis

		icount = 0
		for ishell in range(basis.nshell()):
			aShell = basis.shell(ishell)
			aNBF   = (aShell.am+1)*(aShell.am+2)//2

			for i in range(aNBF):
				jcount = 0

				for jshell in range(basis.nshell()):
					bShell = basis.shell(jshell)
					bNBF = (bShell.am+1)*(bShell.am+2)//2

					for j in range(bNBF):

						for aprimitive in range(aShell.nprimitive):
							for bprimitive in range(bShell.nprimitive):

								alphaA = aShell.exp(aprimitive)
								alphaB = bShell.exp(bprimitive)
								alpha  = alphaA + alphaB
								zeta   = alphaA*alphaB/(alphaA+alphaB)

								A      = np.array( [mol.x(aShell.ncenter), mol.y(aShell.ncenter),mol.z(aShell.ncenter)] )
								B      = np.array( [mol.x(bShell.ncenter), mol.y(bShell.ncenter),mol.z(bShell.ncenter)] )
								P      = (alphaA*A + alphaB*B)/(alphaA+alphaB)

								Ca     = aShell.coef(aprimitive)
								Cb     = bShell.coef(bprimitive)

								kappa  = np.exp( -zeta*np.linalg.norm(A-B)**2 )
								KK     = kappa*(np.pi/alpha)**(3/2)

								XYZ    = self.osRecursion(P-A,P-B,alpha,aShell.am,bShell.am)
								x,y,z  = XYZ[0], XYZ[1], XYZ[2]

								aIndices = self.fillIndices(aShell.am)[i]
								bIndices = self.fillIndices(bShell.am)[j]
								l1, l2   = aIndices[0], bIndices[0]
								m1, m2   = aIndices[1], bIndices[1]
								n1, n2   = aIndices[2], bIndices[2]

								self.S[icount,jcount] += KK*Ca*Cb*x[l1,l2]*y[m1,m2]*z[n1,n2]

						jcount += 1
				icount += 1

		return self.S


	def computeDipole(self):

		basis  = self.basis

		##  converge uhf density
		self.uhf.computeEnergy()

		icount = 0
		for ishell in range(basis.nshell()):
			aShell = basis.shell(ishell)
			aNBF   = (aShell.am+1)*(aShell.am+2)//2

			for i in range(aNBF):
				jcount = 0

				for jshell in range(basis.nshell()):
					bShell = basis.shell(jshell)
					bNBF = (bShell.am+1)*(bShell.am+2)//2

					for j in range(bNBF):

						for aprimitive in range(aShell.nprimitive):
							for bprimitive in range(bShell.nprimitive):

								alphaA = aShell.exp(aprimitive)
								alphaB = bShell.exp(bprimitive)
								alpha  = alphaA + alphaB
								zeta   = alphaA*alphaB/(alphaA+alphaB)

								A      = np.array( [mol.x(aShell.ncenter), mol.y(aShell.ncenter),mol.z(aShell.ncenter)] )
								B      = np.array( [mol.x(bShell.ncenter), mol.y(bShell.ncenter),mol.z(bShell.ncenter)] )
								P      = (alphaA*A + alphaB*B)/(alphaA+alphaB)

								Ca     = aShell.coef(aprimitive)
								Cb     = bShell.coef(bprimitive)

								kappa  = np.exp( -zeta*np.linalg.norm(A-B)**2 )
								KK     = kappa*(np.pi/alpha)**(3/2)

								XYZ    = self.osRecursion(P-A,P-B,alpha,aShell.am+1,bShell.am+1)
								x,y,z  = XYZ[0], XYZ[1], XYZ[2]

								aIndices = self.fillIndices(aShell.am)[i]
								bIndices = self.fillIndices(bShell.am)[j]
								l1, l2   = aIndices[0], bIndices[0]
								m1, m2   = aIndices[1], bIndices[1]
								n1, n2   = aIndices[2], bIndices[2]

								self.dx[icount,jcount] += KK*Ca*Cb*y[m1,m2]*z[n1,n2]*(x[l1+1,l2] + A[0]*x[l1,l2])
								self.dy[icount,jcount] += KK*Ca*Cb*x[l1,l2]*z[n1,n2]*(y[m1+1,m2] + A[1]*y[m1,m2])
								self.dz[icount,jcount] += KK*Ca*Cb*x[l1,l2]*y[m1,m2]*(z[n1+1,n2] + A[2]*z[n1,n2])

						jcount += 1
				icount += 1

		nuclearDipole = psi4.core.nuclear_dipole(self.mol).to_array()

		MU  = [0,0,0]
		dAB = self.uhf.Da + self.uhf.Db

		for k,matrix in enumerate( [self.dx, self.dy, self.dz] ):
			MU[k] += -np.einsum( "mn,mn", dAB, matrix ) + nuclearDipole[k]

		return np.linalg.norm(MU)


	def osRecursion( self, PA, PB, alpha, AMa, AMb ):

		xyz   = []
		for i in range(3):
			I = np.zeros((AMa+1, AMb+1))
			I[0,0] = 1.0

			for a in range(1,AMa+1):
				I[a,0] = PA[i]*I[a-1,0]
				if a >= 2:
					I[a,0] += (a-1)/(2*alpha)*I[a-2,0] 

			for b in range(1,AMb+1):
				I[0,b] = PB[i]*I[0,b-1]
				if b >= 2:
					I[0,b] += (b-1)/(2*alpha)*I[0,b-2]

			for a in range(1,AMa+1):
				for b in range(1,AMb+1):
					I[a,b] = PB[i]*I[a,b-1] + a/(2*alpha)*I[a-1,b-1] 
					if b >= 2:
						I[a,b] +=  (b-1)/(2*alpha)*I[a,b-2]

			xyz.append(I)

		return xyz



if __name__ == '__main__':
	config = configparser.ConfigParser()
	config.read('Options.ini')

	mol   = psi4.geometry(config['DEFAULT']['molecule'])
	basis = psi4.core.BasisSet.build(mol, 'BASIS', config['DEFAULT']['basis'], puream=0)
	mints = psi4.core.MintsHelper(basis)

	uhf       = UHF(mol,mints)
	integrals = OS(basis,mol,uhf)

	print( integrals.computeOverlap() - uhf.S)
	print( integrals.computeKinetic() - uhf.T )
	print( "Dipole moment= ", integrals.computeDipole() )
