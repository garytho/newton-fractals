import matplotlib.pyplot as plt
import numpy as np
import math

#this class defines the polynomial with a given degree and random roots within a given root range
class poly:
	def __init__(self, degree = 3, root_range = (-0.8412, 0.8813), dim = 100, tol = 0.01, cmap = 'viridis', seed = 80085):
		"""
		degree: (int d) degree of polynomial
		root_range: (int a, int b) complex roots c will be chosen so that abs(real(c)) and abs(im(c)) are within root_range
		dim: (int dim) number of different coordinates for starting points of newton method (dim^2 points queried total)
		tol: (int tol) tolerance for newton's method
		cmap: pyplot colormap layout
		seed: integer useful for keeping track of cool fractals
		"""
		self.degree = degree
		self.root_range = root_range
		self.tol = tol
		self.seed = seed
		np.random.seed(self.seed)

		self.roots = np.random.rand(degree) * (self.root_range[1] - self.root_range[0]) + self.root_range[0] + ( np.random.rand(degree) * (self.root_range[1] - self.root_range[0]) + self.root_range[0] ) * (0+1j) 

		#generates coefficients of polynomial with the given random roots
		self.coeff = np.poly(self.roots)

		self.dim = dim
		self.root_array = []

		#the distance between each query position of the newton method
		self.dim_inc = abs(self.root_range[1] - self.root_range[0]) / self.dim


	def f(self, a):
		#calculates the value of the polynomial f at point a
		x_vals = np.asarray([ a ** k for k in reversed(range(0, self.degree + 1))])
		output = np.dot(self.coeff, x_vals)
		return output

	def d_f_inv(self, a):
		#calculates the inverse of the derivative of polynomial f at a 
		x_vals = np.asarray([k * (a ** (k - 1)) for k in reversed(range(0, self.degree + 1))])
		x_vals[-1]=0
		output = 1 / np.dot( self.coeff, x_vals)
		return output

	def newton(self, a0, tol):
		#iterates newton's method until the change in updates is below tol

		change = -1
		a = a0
		while change >= tol or change == -1:
			a1 = a - self.d_f_inv(a) * self.f(a)
			change = abs(a1 - a)
			a = a1
		return a

	def get_val(self, end_point):
		#returns the index of the root closest to the value end_point that newton's method converged to
		idx = (np.abs(self.roots - end_point)).argmin()
		return idx


	def generate_map(self):
		#generates the newton method results for every query position		
		for i in range( self.dim ):
			print(i)
			root_map = []
			for j in range( self.dim ):
				#this line runs newton method at position (i, j) of domain
				root_map.append(self.get_val( self.newton(self.root_range[0] + i*self.dim_inc + self.root_range[0]*1j + j*self.dim_inc*1j, self.tol) ))

			root_map = np.asarray([root_map])
			
			if i == 0:
				self.root_array = root_map
			else:
				self.root_array = np.concatenate((self.root_array, root_map))
				

		return self.root_array


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-deg", "--degree", type=int, default = 3, help = "the degree of the polynomial")
parser.add_argument("-r", "--rootrange", type=tuple, default = (-0.8412, 0.8813), help="the range of values of coordinates of roots of the polynomial in the complex plane")
parser.add_argument("-dim", "--dimension", type=int, default = 300, help="number of different query positions tested")
parser.add_argument("-tol", "--tolerance", type=float, default = 0.01, help="tolerance for newton method")
parser.add_argument("-cmap", "--colormap", default = "viridis", help="pyplot color scheme")
parser.add_argument("-s", "--seed", type=int, default = np.random.randint(0, 1000000), help="seed (useful for recreating specific fractal)")

args = parser.parse_args()


test_poly = poly(args.degree, args.rootrange, args.dimension, args.tolerance, args.colormap, args.seed)

a = test_poly.generate_map()

fig, ax = plt.subplots()

ax.imshow(a, cmap=args.colormap, interpolation='nearest')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])

plt.show()
