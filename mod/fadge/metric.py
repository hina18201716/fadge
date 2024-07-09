# Copyright (C) 2020 Chi-kwan Chan
# Copyright (C) 2020 Steward Observatory
#
# This file is part of fadge.
#
# Fadge is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Fadge is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with fadge.  If not, see <http://www.gnu.org/licenses/>.


from jax import numpy as np, jit


def Cartesian(ndim=3, **kwargs):

    assert ndim > 0

    g = np.identity(ndim, **kwargs) # render constant metric

    def metric(x): # closure on `g`
        return g

    return metric


def Minkowski(aspin=0.0,ndim=4, **kwargs):

    assert ndim > 1

    g = np.diag(np.array([-1.0] + [1.0] * (ndim-1), **kwargs)) # render constant metric

    def metric(x): # closure on `g`
        return g

    return metric


def KerrSchild(aspin=0.0, ndim=4, **kwargs):

    assert ndim == 4

    eta = Minkowski(ndim)(None)
    aa  = aspin * aspin

    @jit
    def metric(x): # closure on `eta`, `aspin`, and `aa`
        zz = x[3] * x[3]
        kk = 0.5 * (x[1] * x[1] + x[2] * x[2] + zz - aa)
        rr = np.sqrt(kk * kk + aa * zz) + kk
        r  = np.sqrt(rr)
        f  = (2.0 * rr * r) / (rr * rr + aa * zz)
        l  = np.array([
            1.0,
            (r * x[1] + aspin * x[2]) / (rr + aa),
            (r * x[2] - aspin * x[1]) / (rr + aa),
            x[3] / r,
        ])
        return eta + f * l[:,np.newaxis] * l[np.newaxis,:]

    return metric


def SpericalKS(aspin=0.0, ndim=4, **kwargs):

    assert ndim == 4
    aa  = aspin * aspin

    @jit
    def metric(x): 

        r = x[1]
        theta = x[2]
        phi = x[3]
        rr = r * r
        f = 2 * r / ( rr + aa * np.cos(theta) * np.cos(theta))
        l  = np.array([
            1.0,
            0.0,
            0.0,
            aa * np.sin(theta)**2,
        ])

        g0 = np.zeros((4, 4), **kwargs)
        g0 = g0.at[0,0].set(-1)
        g0 = g0.at[0,1].set(1)
        g0 = g0.at[2,2].set(rr + aa * np.cos(theta) * np.cos(theta) )
        g0 = g0.at[3,3].set(( rr + aa ) * np.cos(theta) * np.cos(theta) )
        g0 = g0.at[1,3].set( aspin * np.sin(theta) * np.sin(theta) )
        g0 = g0.at[3,1].set(g0[1,3])
        g0 = g0.at[1,0].set(g0[0,1])
        
        return g0 

    return metric


def BoyerLindquist(aspin=0.0, ndim = 4 ,**kwargs):
    
    assert ndim == 4
    aa = aspin * aspin

    @jit
    def metric(x): 
        
        t = x[0]
        r = x[1]
        theta = x[2]
        phi = x[3]
        
        rr = r * r 
        lamda = 0.00  # deviation parameter 
        
        P = rr + aa * np.cos(theta) * np.cos(theta) 
        Q = rr + aa - 2 * (1 + lamda * lamda ) * r 

        g = np.zeros((4, 4), **kwargs)

        
        g[0][0] =  -1 + 2*r/P           
        g[1][1] = P/Q
        g[2][2] = P
        g[3][3] = (rr + aa + 2 * aa * r * np.sin(theta) * np.sin(theta) / P ) * np.sin(theta) ** 2
        
        g[0][3] = - 4 * aspin * r * np.sin(theta) * np.sin(theta) / P
        g[3][0] = g[0][3]
        
        return g 
        
    # # BL to Cartesian KS
    # h = np.sqrt( 1 - aa )
    # L = np.ln( r)
    # x[0] = s[0] + 
    
    return metric