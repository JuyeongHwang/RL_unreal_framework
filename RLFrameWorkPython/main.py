import numpy as np


a = np.array([[[1,3,2],[3,2,4],[2,4,5]],[[2,4,3],[2,1,3],[1,2,3]]])

print("(A) : ", a[:,0]*np.array([1,0,1]))

reference = np.array([[[0,0,0]],[[0,0,0]]])

a2 = np.concatenate([reference,a],axis=1)
print("(B) : ",a2)

