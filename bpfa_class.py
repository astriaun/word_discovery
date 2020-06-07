#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import svds

class BPFA():

	# X = H (Z * W)

	def __init__(self, X, K=100, min_iter=200, max_iter=1000, tol=10**(-3), init=None):
		"""
		Parameters:
			X			-- transition count matrix matrix of size [NxM]
			K			-- number of expected features, to be inferrred
			min_iter	-- minimum number of iterations
			max_iter 	-- maximum number of iterations
			tol			-- tolerance for the stopping criterion
			init 		-- optional initialization of H and W,
							init = [H_init, W_init]
		"""

		self.X 			= X
		self.K 			= K
		self.min_iter 	= min_iter
		self.max_iter 	= max_iter
		self.tol 		= tol
		self.init 		= init

	def init_WH(self, X, K):
		"""
		Initialize the matrices H and W using SVD for non-negative matrices
		"""
		N, M = X.shape
		H = np.zeros([N, K])
		W = np.zeros([K, M])
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
			H[:, j] = np.sqrt(S[j] * sigma) * u
			W[j, :] = np.sqrt(S[j] * sigma) * v

		return(H, W)

	def initialize(self):
		"""
		Initialize the matrices H, W, and Z
		"""
		N, M = self.X.shape
		K = self.K

		#initialize W,H with nndsvd
		if self.init == None:
			H, W = self.init_WH(self.X, K)
		else:
			H = self.init[0]
			W = self.init[1]

		K = self.K = np.sum( W.sum(axis=0) > 0 )
		self.H = H[:, :K]
		self.W = W[:K, :]

		#initialize Z
		self.Z = np.ones([K, M])
		self.ZW = self.Z*self.W

		#initialize pi
		self.a = self.b = 1
		self.alpha = self.a/K
		self.beta = self.b*(K-1)/K
		self.pi = 10**(-6) * np.ones(K)

		#initialize sigma_H, covariance for H
		self.c = 1
		self.d = 10**(-6)

		#initialize sigma_E, additive noise
		self.g = self.h = 10**(-6)

		self.sigma_H = np.ones([N, K])
		self.sigma_E = np.ones(N)

		self.E = self.X - self.H.dot(self.ZW)
		self.init_error = self.error()
		print("initialized, se =", self.init_error)


	def train(self):
		"""
		Training the model. Returns H, Z, W, number of iterations, process in norm
		"""
		N, M = self.X.shape
		K = self.K
		alpha, beta = self.alpha, self.beta
		process = []
		n_iter = 0	
		while True:
			n_iter += 1
			print('--- iteration nr', n_iter, '---')

			self.Z = self.update_Z()
			Zcnt = np.sum(self.Z.sum(axis=1) > self.eps)
			if  Zcnt < K:
				K = self.K = Zcnt
				K_nz = self.Z.sum(axis=1) > self.eps
				self.Z = self.Z[K_nz, :]
				self.W = self.W[K_nz, :]
				self.H = self.H[:, K_nz]
				self.sigma_H = self.sigma_H[:, K_nz]
				alpha = self.a/K
				beta = self.b*(K-1)/K

			print('# non-zero columns of Z::', Zcnt, 'sum(Z) =', self.Z.sum()/Zcnt)

			self.ZW = self.Z*self.W

			self.pi = np.random.beta(alpha + self.Z.sum(axis=1), M + beta - self.Z.sum(axis=1))


			self.W = self.update_W()
			self.ZW = self.Z*self.W

			self.H = self.update_H()

			self.E = self.X - self.H.dot(self.ZW)


			c = self.c + 0.5
			d = self.d + 0.5*self.H**2

			g = self.g + 0.5*M
			h = self.h + 0.5*(self.E**2).sum(axis=1)

			self.sigma_H = np.random.gamma(c, 1/d)
			self.sigma_E = np.random.gamma(g, 1/h)

			error = self.error()
			process.append(error)
			self.print_error(error)


			if n_iter % 10 == 0 and n_iter >= self.min_iter:
				if np.abs(process[-10] - error) / self.init_error < self.tol :
					break

			if n_iter == self.max_iter:
				break

		return(self.H, self.Z, self.W, n_iter, process)

	def update_H(self):
		"""
		Update of H
		"""
		H = self.H.copy()
		N, K = H.shape

		ZWZW = self.ZW.dot(self.ZW.T)
		XZW = self.X.dot(self.ZW.T)
		for n in range(N):
			var = np.linalg.cholesky(ZWZW*self.sigma_E[n] + np.diag(self.sigma_H[n, :])).T
			mean = self.sigma_E[n] * np.linalg.solve(var.T, XZW[n, :].T)
			H[n, :] = np.abs( np.linalg.solve(var, np.random.randn(K) + mean) )
		return(H)

	def update_W(self):
		"""
		Update of W
		"""
		W = self.W.copy()
		K, M = W.shape

		Sigma_E = np.diag(self.sigma_E)
		HSig = self.H.T.dot(Sigma_E)
		HTH = HSig.dot(self.H)
		for m in range(M):
			HSigZ = (HSig.T*self.Z[:,n]).T
			Zn = self.Z[:, n].reshape([K, 1])
			ZZ = Zn.dot(Zn.T)
			var = np.linalg.cholesky(HTH*ZZ + np.eye(K)).T
			var = np.linalg.solve(var, np.eye(K))
			var = var.dot(var.T)
			mean = var.dot( HSigZ.dot(self.X[:, n]) )
			W[:, n] = np.abs( np.random.multivariate_normal(mean, var) )
		return(W)

	def update_Z(self):
		"""
		Update of Z
		"""
		Z = self.Z.copy()
		K, M = Z.shape
		
		Sigma_E = np.diag(self.sigma_E)
		HSig = self.H.T.dot(Sigma_E)
		HSigX = HSig.dot(self.X)
		HTH = HSig.dot(self.H)
		for k in range(K):
			tmpZW = self.ZW
			tmpZW[k, :] = 0
			res = HTH[k,k]*self.W[k, :]**2 - 2*self.W[k, :]*(HSigX[k, :] - HTH[k,:].dot(tmpZW))
			t1 = np.log(1-self.pi[k]) - np.log(self.pi[k]) + 0.5*res
			prob = 1/(1+np.exp(t1))
			Z[k, :] = np.random.binomial(1, prob)
		return(Z)

	def error(self):
		return( norm(self.E) )

	def print_error(self, error=None):
		if error != None:	print('--- error =', error, '---')
		else:				print('--- error =', self.error(), '---')	


def pos(vector):
	return( np.multiply(vector>=0, vector) )

def neg(vector):
	return( np.multiply(vector<0, -vector) )


