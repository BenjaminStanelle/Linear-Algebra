"""
    Fri Dec 21 13:54:08 CST 2018

    some useful links
      https://docs.scipy.org/doc/numpy/reference/routines.linalg.html
      https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html
"""
"""
    Benjamin Stanelle 
"""
import numpy as np

def fun06(C, d) :
    """
        your code goes in this function

        DO NOT PRINT ANYTHING FROM THE FUNCTION
    """
    A = np.array([C[0][2:5], C[1][2:5], C[2][2:5]])
    b = np.array([d[len(d)-3:len(d)]]).T 
    
    I = np.identity(len(A), dtype = int)
    x = np.matrix(A).I*b
    µ = sum(np.diagonal(A))/len(A)
    Q = A-µ*I
    y = np.matrix(Q).I*b 

    normXY = np.linalg.norm(x)-np.linalg.norm(y)

    return x, A, b, Q, y, normXY


#
#  DO NOT MODIFY ANYTHING BELOW THIS
#
##################  main  ##################
if __name__ == "__main__":
    C = np.array( [[ 8, 13, 16, 50, 45, 83],
                   [ 5, 57, 79, 26, 60, 54],
                   [53, 47, 31, 65, 23, 70],
                   [78,  1, 53, 69, 91,  8],
                   [93, 34, 17, 75, 15, 44]] )
     
    d = np.array( [100, 200, 300, 209, 68, 229] )
    
    
    x, A, b, Q, y, normXY = fun06(C, d)
    
    print( "A:", A )
    print( "\nb:", b )
    print( "\nx: ", x)
    
    print( "\nQ:", Q )
    print( "\ny:", y)
    
    print( "\n||x - y||: ", normXY)
