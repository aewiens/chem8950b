import numpy as np


class amIndices(object):

	def __init__(self,L):

		self.L     = L
		self.count = 0
		self.maxit = (L+1)*(L+2)//2

		self.indices = np.zeros((self.maxit,3))

		##  fill in index list
		self.passToRight([self.L, 0, 0],0)


	def passToRight(self,row,index):

		if self.count < self.maxit:
			newList = row[:]
			self.count +=1 
			self.indices[self.count-1] = newList

			for i in range( index, 2 ):
				amnt = newList[i]

				for j in range( 0, amnt ):
					if newList[i] > 0:
						newList[i] -= 1
						newList[i+1] += 1
						self.passToRight( newList, i + 1 )	



if __name__ == '__main__':
	test = amIndices(2)
	for i in test.indices:
		print(i)





