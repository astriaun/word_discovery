#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import svds

class NNMF():

	# X = W H

	def __init__(self, X, K, max_iter=200, tol=10**(-3), cf=0, init=None):
		"""
		Parameters:
			X 			-- transition count matrix of size [MxN]
			K			-- number of expected features
			max_iter 	-- maximum number of iterations
			tol			-- tolerance for the stopping criterion
			cf 			-- cost function used
							0 - Kullback-Leibler Divergence
							1 - Euclidean distance ( squared Frobenius norm )
			init 		-- optional initialization of H and W,
							init = [W_init, H_init]
		"""
		self.X 			= X
		self.K 			= K
		self.max_iter 	= max_iter
		self.tol 		= tol
		self.cf 		= cf
		self.init 		= init

	def init_WH(self, X, K):
		"""
		Initialize the matrices W and H using SVD for non-negative matrices
		"""
		M, N = X.shape
		W = np.zeros([M, K])
		H = np.zeros([K, N])
		U, S, V = svds(X, K)
		for j in range(K):
			x = U[:, j]
			y = V[j, :]
			xp = pos(x)
			xn = neg(x)
			yp = pos(y)
			yn = neg(y)
			mp = norm(xp) * norm(yp)
			mn = norm(xn) * norm(yn)
			if mp > mn:
				u = xp/norm(xp)
				v = yp/norm(yp)
				sigma = mp
			else:
				u = xn/norm(xn)
				v = yn/norm(yn)
				sigma = mn
			W[:, j] = np.sqrt(S[j] * sigma) * u
			H[j, :] = np.sqrt(S[j] * sigma) * v

		return(W, H)

	def initialize(self):
		"""
		Initialize the matrices W, H
		"""
		M, N = self.X.shape
		K = self.K

		if self.init == None:
			W, H = self.init_WH(self.X, K)
		else:
			W = self.init[0]
			H = self.init[1]

		S = np.diag( 1/W.sum(axis=0) )
		self.W = np.matmul(W, S)
		self.H = np.matmul(np.linalg.inv(S), H)

		self.init_error = self.error()
		print("initialized, error = ", self.init_error)

	def train(self):
		"""
		Training the model. Returns W, H, number of iterations, process in cost function, process in norm
		"""
		process_norm = []
		process_cf = []
		prev_error = self.init_error
		for n_iter in range(self.max_iter):
			
			delta_W = self.update_W()
			self.W *= delta_W
			self.W /= self.W.sum(axis=0)

			delta_H = self.update_H()
			self.H *= delta_H
						
			process_norm.append(norm(self.X - self.W.dot(self.H)))
			process_cf.append(self.error())
			
			if n_iter % 10 == 0 :
				print('--- iteration nr', n_iter, '---')
				error = self.error()
				self.print_error(error)
				if np.abs(prev_error - error) / self.init_error < self.tol :
					break
				prev_error = error

		return(self.W, self.H, n_iter, process_norm, process_cf)


	def update_H(self):
		"""
		Update of H
		"""
		if (self.cf == 0): 
			W_sum = np.sum(self.W, axis=0)
			WH_data = self.W.dot(self.H)
			div = np.divide( self.X, WH_data, out=np.ones_like(self.X), where=WH_data!=0 )
			numerator = self.W.T.dot(div)
			denominator = W_sum[:, np.newaxis]
		else:
			numerator = np.dot(self.W.T, self.X)
			denominator = np.dot(np.dot(self.W.T, self.W), self.H)
		delta_H = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator!=0)
		return(delta_H)

	def update_W(self):
		"""
		Update of W
		"""
		if (self.cf == 0): 
			H_sum = np.sum(self.H, axis=1)
			WH_data = self.W.dot(self.H)
			div = np.divide( self.X, WH_data, out=np.ones_like(self.X), where=WH_data!=0 )
			numerator = div.dot(self.H.T)
			denominator = H_sum[np.newaxis, :]
		else:
			numerator = np.dot(self.X, self.H.T)
			denominator = np.dot(self.W, np.dot(self.H, self.H.T))
		delta_W = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator!=0)
		return(delta_W)

	def error(self,):
		if(self.cf == 0):
			return(self.kld())
		else:
			return(self.fro())

	def kld(self): # Kullback-Leibler Divergence
		WH_data = self.W.dot(self.H)
		indices = self.X > 0
		X_data = self.X[indices]
		WH_data = WH_data[indices]
		div = np.divide(X_data, WH_data)
		sum_WH = np.dot(np.sum(self.W, axis=0), np.sum(self.H, axis=1))
		res = np.dot(X_data, np.log(div))
		res += sum_WH - X_data.sum()
		return(res)

	def fro(self): #squared Frobenius norm
		res = ((self.X - np.dot(self.W, self.H))**2).sum()
		return(res)

	def print_error(self, error=None):
		if error != None:	print('--- error =', error, '---')
		else:				print('--- error =', self.error(), '---')	


def pos(vector):
	return( np.multiply(vector>=0, vector) )

def neg(vector):
	return( np.multiply(vector<0, -vector) )

