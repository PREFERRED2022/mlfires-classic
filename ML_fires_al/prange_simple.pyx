from cython.parallel import prange
cdef int i
cdef int n = 1000000000
cdef long int sum = 0

for i in prange(n, nogil=True):
    sum += i

print(sum)