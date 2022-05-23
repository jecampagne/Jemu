
from typing import Iterable, Optional

import jax
from jax import vmap, jit
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


Array = jnp.array

# JEC 23/5/2022 bspline & bilinear (edges)

@jit
def jec_interp2d_v2(xnew,ynew,xp,yp,zp):
    """
    (xnew,ynew): two 1D vector  of same size where to perform predictions  f(xnew[i],ynew[i])
    (xp,yp): original grid points 1D vector
    zp: original values of functions  zp[i,j] = value at xp[i], yp[j]
    """
    
    
    M = 1./16 * jnp.array([[0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                           [0, 0, 0, 0, -8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                           [0, 0, 0, 0, 16, -40, 32, -8, 0, 0, 0, 0, 0, 0, 0, 0], 
                           [0, 0, 0, 0, -8, 24, -24, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, -8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0], 
                           [4, 0, -4, 0, 0, 0, 0, 0, -4, 0, 4, 0, 0, 0, 0, 0], 
                           [-8, 20, -16, 4, 0, 0, 0, 0, 8, -20, 16, -4, 0, 0, 0, 0],
                           [4, -12, 12, -4, 0, 0, 0, 0, -4, 12, -12, 4, 0, 0, 0, 0],
                           [0, 16, 0, 0, 0, -40, 0, 0, 0, 32, 0, 0, 0, -8, 0, 0], 
                           [-8, 0, 8, 0, 20, 0, -20, 0, -16, 0, 16, 0, 4, 0, -4, 0], 
                           [16, -40, 32, -8, -40, 100, -80, 20, 32, -80, 64, -16, -8, 20, -16, 4], 
                           [-8, 24, -24, 8, 20, -60, 60, -20, -16, 48, -48, 16, 4, -12, 12, -4], 
                           [0, -8, 0, 0, 0, 24, 0, 0, 0, -24, 0, 0, 0, 8, 0, 0], 
                           [4, 0, -4, 0, -12, 0, 12, 0, 12, 0, -12, 0, -4, 0, 4, 0], 
                           [-8, 20, -16, 4, 24, -60, 48, -12, -24, 60, -48, 12, 8, -20, 16, -4], 
                           [4, -12, 12, -4, -12, 36, -36, 12, 12, -36, 36, -12, -4, 12, -12, 4]]
                         )
    
    M1 = jnp.array([[1.,0.,0.,0.],
                    [-1.,1.,0.,0.],
                    [-1.,0.,1.,0.],
                    [1.,-1.,-1.,1.]])

    def built_Ivec(zp,ix,iy):
        return jnp.array([zp[ix+i,iy+j] for j in range(-1,3) for i in range(-1,3)])


    def built_Ivec1(zp,ix,iy):
        return jnp.array([zp[ix+i,iy+j] for j in range(0,2) for i in range(0,2)])

    
    
    def compute_basis(x,order=3):
        """
        x in [0,1]
        """
        ######return  jnp.array([1.0]+[x**i for i in jnp.arange(1, order + 1)])[:,jnp.newaxis]    
        return jnp.array([x**i for i in jnp.arange(0, order+1)])
    
    def tval(xnew,ix,xp):
        return (xnew-xp[ix-1])/(xp[ix]-xp[ix-1])
    
    ix = jnp.clip(jnp.searchsorted(xp, xnew, side="right"), 0, len(xp)-1)
    iy = jnp.clip(jnp.searchsorted(yp, ynew, side="right"), 0, len(yp)-1)

    
    def bilinear_interp(ix,iy):
        Iv = built_Ivec1(zp,ix-1,iy-1)
        av = M1 @ Iv
        amtx = av.reshape(2,2,-1)
        tx = tval(xnew,ix,xp)
        ty = tval(ynew,iy,yp)
        basis_x = compute_basis(tx,order=1)
        basis_y = compute_basis(ty,order=1)
        res = jnp.einsum("i...,ij...,j...",basis_y,amtx,basis_x)
        return res

    def bispline_interp(ix,iy):
        Iv = built_Ivec(zp,ix-1,iy-1)
        av = M @ Iv
        amtx = av.reshape(4,4,-1)
        tx = tval(xnew,ix,xp)
        ty = tval(ynew,iy,yp)
        basis_x = compute_basis(tx)
        basis_y = compute_basis(ty)
        res = jnp.einsum("i...,ij...,j...",basis_y,amtx,basis_x)
        return res
    
    condx = jnp.logical_and(ix>=2, ix<=len(xp)-2)
    condy = jnp.logical_and(iy>=2, iy<=len(yp)-2)
    
    cond = jnp.logical_and(condx,condy)
    return jnp.where(cond,
             bispline_interp(ix,iy),
             bilinear_interp(ix,iy))

##JEC 13/5/2022 taken from https://github.com/adam-coogan/jaxinterp2d
## very simple, no bound chek

def coogan_interp2d(
    x: Array,
    y: Array,
    xp: Array,
    yp: Array,
    zp: Array,
    fill_value: Optional[Array] = None,
) -> Array:
    """
    Bilinear interpolation on a grid. ``CartesianGrid`` is much faster if the data
    lies on a regular grid.
    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`
    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
    """
    if xp.ndim != 1 or yp.ndim != 1:
        raise ValueError("xp and yp must be 1D arrays")
    if zp.shape != (xp.shape + yp.shape):
        raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    #print("ix,iy:",ix,iy)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    #print("Z_ij:",z_11,z_21,z_12,z_22)
    
    
    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = jnp.logical_or(
            x < xp[0], jnp.logical_or(x > xp[-1], jnp.logical_or(y < yp[0], y > yp[-1]))
        )
        z = jnp.where(oob, fill_value, z)

    return z

#interp2d = coogan_interp2d
interp2d = jec_interp2d_v2
