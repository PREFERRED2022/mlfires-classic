import numpy as np
cimport numpy as cnp
import cython
from cython.parallel import prange
INT32 = np.int32
ctypedef cnp.int32_t INT32_t
from cython cimport view

cnp.import_array()

cdef long* get_grid_xy(long _id, long firstid, long rdiff) nogil:
    cdef long[2] res
    res[0] = int((_id-firstid)/rdiff) #row
    res[1] = _id-firstid-rdiff*res[0] #col
    return res

cdef long get_grid_x(long _id, long firstid, long rdiff, long row) nogil:
    cdef int col
    col = _id-firstid-rdiff*row #col
    return col

cdef long get_grid_y(long _id, long firstid, long rdiff) nogil:
    cdef int row
    row = int((_id-firstid)/rdiff) #row
    return row


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def npapplypar(int nt, cnp.ndarray[INT32_t, ndim=1] a, long firstid, long rdiff):
    cdef int i, M, res
    M = a.shape[0]
    ares = view.array(shape=(M,2), itemsize=sizeof(int), format="i")
    cdef int [:, :] ares_view = ares
    cdef int [:] a_view = a
    for i in prange(M, num_threads=nt, nogil=True):
        ares_view[i,0]=get_grid_y(a_view[i], firstid, rdiff)
        ares_view[i,1]=get_grid_x(a_view[i], firstid, rdiff, ares_view[i,0])
    arespy=np.asarray(ares_view)
    return arespy

'''
def npapplypar(int nt, func, cnp.ndarray[INT32_t, ndim=1] a, *argv):
    cdef int i, M, res
    M = a.shape[0]
    ares = view.array(shape=(M), itemsize=sizeof(int), format="i")
    for i in prange(M, num_threads=nt, nogil=True):
        with gil:
            ares[i]=func(a[i],*argv)
    return ares
'''
