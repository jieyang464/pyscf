#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Providers for the AO derivative integrals (nabla i,j|k,l) that the correlated
nuclear-gradient kernels contract block by block.

Those kernels accept one of these as a ``deriv_eri`` argument and call it where
they would otherwise call ``mol.intor``, so how a block is obtained is separated
from what is done with it.  The default provider, :func:`int2e_ip1`, is the
integral-direct evaluation they have always used, and nothing changes unless a
caller passes something else.
'''

from pyscf import lib
from pyscf.lib import logger


def int2e_ip1(mol, shls_slice=None):
    '''The (nabla i,j|k,l) block selected by ``shls_slice``, ket packed as s2kl.

    Evaluated on the fly and thrown away by the caller, which is what makes the
    derivative ERI affordable: only one block of it exists at a time.  With
    ``shls_slice=None`` the whole thing comes back instead, shape
    ``(3,nao,nao,nao_pair)`` -- the same call, a wider slice.
    '''
    return mol.intor('int2e_ip1', comp=3, aosym='s2kl', shls_slice=shls_slice)


def deriv_eri_kwargs (deriv_eri):
    '''Forward a provider only when one was given.

    pyscf.df.grad.sacasscf swaps its own Lagrange kernels in over the ones in
    pyscf.grad.sacasscf, and those never evaluate (nabla i,j|k,l).  Passing
    ``deriv_eri=None`` down to them would be a TypeError on a call that opted
    into nothing, so the keyword is only added once there is something to say.
    '''
    return {} if deriv_eri is None else {'deriv_eri': deriv_eri}


class DerivativeERICache:
    '''A drop-in for :func:`int2e_ip1` that evaluates (nabla i,j|k,l) once and
    then serves blocks out of it, so several gradients and NACs at one geometry
    share one evaluation::

        mc_grad = mc.nuc_grad_method (state=0)
        mc_grad.deriv_eri = DerivativeERICache ()
        de_0 = mc_grad.kernel (state=0)   # fills the cache
        de_1 = mc_grad.kernel (state=1)   # warm
        mc_grad.deriv_eri.eri1            # the tensor, kept on the object

    A served block is a view, so serving costs no arithmetic at all; the price
    is paid entirely up front, in memory.  Holding the derivative ERI means
    holding ``3*nao**2*nao_pair*8`` bytes of it -- 20 MB at nao=36, 1.2 GB at
    nao=100, 19 GB at nao=200 -- which is why the kernels stream it by default
    and why this has to be asked for.  Past ``max_memory`` the cache declines to
    build and every call falls through to :func:`int2e_ip1`, so switching reuse
    on can cost time but cannot exhaust memory.

    The tensor is only valid at the geometry it was taken at.  The cache keys on
    the Mole it was built from and rebuilds when handed a different one, so a
    stale cache costs a recomputation rather than a wrong gradient.
    '''

    def __init__(self, max_memory=None):
        self.max_memory = lib.param.MAX_MEMORY if max_memory is None else max_memory
        self.mol = None
        self.eri1 = None      # (3,nao,nao,nao_pair), or None when not cached
        self.ao_loc = None

    def build(self, mol):
        '''Evaluate and keep the whole derivative ERI for ``mol``, unless that
        would exceed ``max_memory``, in which case nothing is kept.'''
        nao = mol.nao_nr()
        nbytes = 3 * nao**2 * (nao*(nao+1)//2) * 8
        self.mol = mol
        self.ao_loc = mol.ao_loc_nr()
        if nbytes > self.max_memory * 1e6:
            logger.debug(mol, 'DerivativeERICache not built: (nabla i,j|k,l) needs '
                         '%.0f MB > max_memory %.0f MB', nbytes/1e6, self.max_memory)
            self.eri1 = None
        else:
            self.eri1 = int2e_ip1(mol)
        return self

    def __call__(self, mol, shls_slice=None):
        if mol is not self.mol:
            self.build(mol)
        # A block can only be cut out of the cached tensor when the ket pair
        # spans the whole basis, which is how the packed index was built.
        if self.eri1 is None or (shls_slice is not None and
                                 tuple(shls_slice[4:]) != (0, mol.nbas, 0, mol.nbas)):
            return int2e_ip1(mol, shls_slice)
        if shls_slice is None:
            return self.eri1
        shl0, shl1, b0, b1 = shls_slice[:4]
        ao_loc = self.ao_loc
        return self.eri1[:, ao_loc[shl0]:ao_loc[shl1], ao_loc[b0]:ao_loc[b1], :]
