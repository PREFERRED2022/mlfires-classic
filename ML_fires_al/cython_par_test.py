#import prange_test
import nppar
import time
import numpy as np

def foo(x, a):
    return x+a

a = np.random.random((50000,500))
b = np.random.random((500,2000))
'''
start=time.time()
c = np.dot(a, b)
print('%.2f s'%(time.time()-start))

start=time.time()
c2 = prange_test.mydot(a, b, 16)
print('%.2f s'%(time.time()-start))
'''
res=nppar.npapplypar(1, np.array([20,32,56,60],dtype=np.int32), 5, 10)
i=1



#print('finished mydot: {} s'.format(time.clock()-t))

#print('Passed test:', np.allclose(c, c2))