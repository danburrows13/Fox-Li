#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 11:54:25 2017

@author: danburrows
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def Calculate_hn(L = 101):
    
    """ Outputs vector h of odd length L := 2N + 1 of ratio hn = fn/gn, with 
    fn and gn Maclaurin coefficients of function f and g, analytic within a 
    nbhd of zero. Here, f(x) = sinc(x) + icosinc(x), 
    g(x) = exp(-x**2) + i2pi**(-1/2)F(x), F(x) Dawson's function. """
    
    n = np.arange(L)
    h = np.divide(sp.special.gamma((n + 2)*0.5), sp.misc.factorial(n + 1))
    
    return h

def Kung(h, B):
    
    """ Given a vector h of length 2N + 1, calculates alpha, gamma vectors of 
    length M, where M = M(tol), such that 
    h[n] approx = sum_m(alpha_m*gamma_m**n) for n = 0, ..., 2N """
    
# =============================================================================
#     Compute SVD of Hankel Matrix
# =============================================================================
    h = h*(B**np.arange(len(h)))
    N = len(h)//2
    U, s, Vh = sp.linalg.svd(sp.linalg.hankel(h[:N + 1], h[N:]),\
                             full_matrices = False)
    
    tol = 2e-15
    sroot = np.sqrt(np.arange(N + 1))*s
    #M = np.unique(sroot[1:] < tol, return_index = True)[1][1]
    M = np.unique(s/s[0] < tol, return_index = True)[1][1]
    
# =============================================================================
#     Calculate Realisation tuple (A, b, c) and alpha, gamma       
# =============================================================================
    Utilde = U[0:N - 1, 0:M]
    Uhat = U[1:N, 0:M]
    S = np.diag(np.sqrt(s[0:M]))
    A = np.matmul(np.linalg.pinv(np.matmul(Utilde, S)), np.matmul(Uhat, S))
    b = np.matmul(S, Vh[0:M, 0])
    c = np.matmul(U[0, 0:M], S)
    gamma, W = np.linalg.eig(A)
    alpha = np.matmul(c, W).T*np.matmul(np.linalg.inv(W), b)
    
    return alpha, gamma/B

def Approx_ind(B):
    
    """ Approximates the indicator function over [-B, B] as a sum of complex
    Gaussians """
    
    x = np.linspace(-(B + 1), B + 1, 6001)
    h = Calculate_hn(101)
    alpha, gamma = Kung(h, B)
    f = lambda x: np.matmul(np.exp(-np.outer(x, np.sqrt(gamma))**2),\
                            alpha).flatten()
    alpha = np.divide(alpha, np.sqrt(np.pi)*gamma)
    gamma = 1/(4.0*gamma**2)
    indtilde = f(x)
    #plt.plot(x, indtilde.real, 'b-', x, abs(x) <= 1, 'y-')
    
    return alpha, gamma

def Approx_mat(mv, nv, omega, B):
    
    """ Calculates matrix Atilde that is used to approximate eigenvalues of 
    Fox-Li operator, and computes the eigenvalues of this matrix, that are 
    themselves the approximations. Here, A.size = mv.size x nv.size. """
    
    h = Calculate_hn(101)
    alpha, gamma = Kung(h, B)
    alpha, gamma = Approx_ind(B)
    Alpha_k = alpha[:,None,None,None]
    Alpha_l = alpha[None,:,None,None]
    Gamma_k = gamma[:,None,None,None]
    Gamma_l = gamma[None,:,None,None]
    M = mv[None,None,:,None]
    N = nv[None,None,None,:]
    
    temp = Alpha_k*Alpha_l*np.exp((np.pi*N)**2/(4.0*(1j*omega - Gamma_k)))\
              *np.sqrt(np.pi/(-(1j*omega - Gamma_k)))*np.sqrt(-np.pi\
               /(1j*omega - Gamma_l + omega**2/(1j*omega - Gamma_k)))\
                *np.exp(-(1j*np.pi*M + np.pi*N*omega/(1j*omega - Gamma_k))**2\
                  /(4.0*(1j*omega - Gamma_l + omega**2/(1j*omega - Gamma_k))))
    
    Atilde = np.sum(temp,(0,1))
    Atilde[np.isnan(Atilde)] = 0.
    plt.imshow(np.abs(Atilde))
    plt.colorbar()
    plt.show()
    
    return Atilde

def ktilde(xi, omega, B):
    
    alpha, gamma = Approx_ind(B)
    beta = np.multiply(alpha, np.sqrt(-np.pi/(1j*omega - gamma)))
    ktilde = np.matmul(np.exp(1/4.*np.outer(xi**2, 1/(1j*omega\
                 - gamma))), beta).flatten()
    
    return ktilde
    
def khat(xi, eps): 
    
    return np.sqrt(np.pi/(eps - 1j))*np.exp(-1/(4.*(eps - 1j))*xi**2).flatten()
    
B = 1 #1e-10 #4.5
omega = 50. 
print([B, omega])
m_max = 2*omega
xi = np.linspace(0, np.min([omega*1.1, 400]), np.min([np.pi*omega, 400*2.]))
mv = xi
nv = mv

Atilde = Approx_mat(mv, nv, omega, B)
eigen = np.linalg.eigvals(Atilde)
eigen_index = np.argsort(np.absolute(eigen))
eigen = eigen[eigen_index][::-1]
ktilde = ktilde(xi, omega, B)

# Plot eigenvalues
plt.plot(eigen.real, eigen.imag, '.')
plt.xlabel(r'Real($\lambda_n$)')
plt.ylabel(r'Imag($\lambda_n$)')
plt.title(r'Approximate Eigenvalues $\{\lambda_n\}$,')
#plt.savefig('omega10.png')
plt.gca().set_aspect('equal')
plt.show()

# Plot ktilde
plt.plot(ktilde.real, ktilde.imag)
plt.xlabel(r'Re$\{\tilde{k}\}$')
plt.ylabel(r'Im$\{\tilde{k}\}$')
#plt.title(r'$\tilde{k}(\xi)$')
plt.gca().set_aspect('equal')
plt.show()
#plt.savefig('ktilde_omega5')


levels = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
plt.subplot(133).contour(np.meshgrid(mv, mv)[0], np.meshgrid(mv, mv)[0].T,\
            abs(Atilde), levels)
plt.gca().set_aspect('equal'); 
#plt.gca().autoscale(tight = True)

  

    
    
    
    
 
