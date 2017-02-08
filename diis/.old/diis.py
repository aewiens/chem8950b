import os, configparser, numpy as np


class DIIS:

	def __init__(self,start,nvector):

		self.diisNvec  = nvector 
		self.start = start 


	def diisP(self,S,focks,densities):

		if len(focks) > self.diisNvec:
			start     = len(focks)-self.diisNvec
			densities = densities[start:len(focks)]
			focks     = focks[start:len(focks)]

		errorVectors = [ F@D@S - S@D@F for F,D in zip(focks,densities) ]

		N = len(errorVectors)
		P = np.zeros(( N+1, N+1))

		for j in range(N):
			P[-1,j] = P[j,-1] = -1
			for k in range(N):
				P[j,k] = np.vdot(errorVectors[j],errorVectors[k])

		return P


	def solveDIIS(self,P,focks):

		if len(focks) > self.diisNvec:
			start     = len(focks)-self.diisNvec
			focks     = focks[start:len(focks)]

		f = np.zeros(( P.shape[0] ))
		f[-1] = -1

		q = np.linalg.solve(P,f)

		return sum( [q[i]*F for i,F in enumerate(focks)] )



