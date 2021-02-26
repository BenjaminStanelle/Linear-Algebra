#Benjamin Stanelle
import numpy as np

def lsSVD(data, tol) :
    A = np.delete(data, 4, axis = 1) 
    B = np.delete(data, (0,1,2,3), axis = 1)
    
    
    U,Σ,V_T = np.linalg.svd(A, full_matrices=1, compute_uv=1)
        #singular 
    r = (Σ >= tol).sum()
    wi = (data[:,0])
    di = (data[:,1])
    hi = (data[:,2])
    pi = (data[:,3])
    x=0
    for i in range(r):
        x1= U.T[i, :] 
        x1 = (np.dot(x1, B)/Σ[i]) * (V_T.T[:, i])
        x += x1
        
    norm = np.linalg.norm(np.dot(A,x))
    return x, Σ, r, norm
 #############################  main  #############################
#
#    DO NOT CHANGE ANYTHING BELOW THIS
#
if __name__ == "__main__" :
    data = np.genfromtxt("waterUsage.csv", dtype=None, delimiter=',', skip_header=5)
    
    # function leastSquaresSvdStudent
    #      parameters:  2D numpy array, int
    #   return values:  1D numpy array, 1D numpy array, float
    #
    
    tolerance = 2.5
    x, S, r, norm = lsSVD(data, tolerance)
    
    print("singular values")
    print(S)
    
    print("\neffective rank = %d when tolerance is %.1f" % (r, tolerance))
    
    print("\nEstimates")
    print("    laundry = %5.1f gallons" % x[0])
    print(" dishwasher = %5.1f gallons" % x[1])
    print("     shower = %5.1f gallons" % x[2])
    print(" sprinklers = %5.1f gallons" % x[3])
    
    print("\nthe norm of the residual is %.1f" % norm)

