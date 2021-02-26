#Benjamin Stanelle
import numpy as np
import matplotlib.pyplot as plt
from numpy  import linalg as LA
def processPoints(ps, pc, points) :
    
    A=np.array([[ps[0][0],ps[1][0],1,0,0,0,0,0,-pc[0][0],0,0,0],
                [0,0,0,ps[0][0],ps[1][0],1,0,0,-pc[1][0],0,0,0],
                [0,0,0,0,0,0,ps[0][0],ps[1][0],-1,0,0,0],
                [ps[0][1],ps[1][1],1,0,0,0,0,0,0,-pc[0][1],0,0],
                [0,0,0,ps[0][1],ps[1][1],1,0,0,0,-pc[1][1],0,0],
                [0,0,0,0,0,0,ps[0][1],ps[1][1],0,-1,0,0],
                [ps[0][2],ps[1][2],1,0,0,0,0,0,0,0,-pc[0][2],0],
                [0,0,0,ps[0][2],ps[1][2],1,0,0,0,0,-pc[1][2],0],
                [0,0,0,0,0,0,ps[0][2],ps[1][2],0,0,-1,0],
                [ps[0][3],ps[1][3],1,0,0,0,0,0,0,0,0,-pc[0][3]],
                [0,0,0,ps[0][3],ps[1][3],1,0,0,0,0,0,-pc[1][3]],
                [0,0,0,0,0,0,ps[0][3],ps[1][3],0,0,0,-1]])
    

    b=np.array([[0],[0],[-1],[0],[0],[-1],[0],[0],[-1],[0],[0],[-1]])
    x=LA.solve(A,b)
    x=np.transpose(x)
    
    p1=x[0][0]
    p2=x[0][1]
    p3=x[0][2]
    p4=x[0][3]
    p5=x[0][4]
    p6=x[0][5]
    p7=x[0][6]
    p8=x[0][7]

    H=np.array([[p1,p2,p3],[p4,p5,p6],[p7,p8,1]])
   
    #matmul is like dot but can't multiply by scalars, stacks of matrices together
    projected=np.matmul(H,points)
    projectedR=projected[:2]/projected[2]
    return H,projectedR


def plotPoints(projected, test_points) :
    #
    #  DO NOT MODIFY THIS SECTION
    #
    
    plt.plot(test_points[0, :], test_points[1, :], '-o')
    plt.title('skewed points')
    # from https://stackoverflow.com/questions/16183462/saving-images-in-python-at-a-very-high-quality
    #plt.savefig('homographySkewed.eps', format='eps', dpi=1000)

    plt.figure(2)
    plt.plot(projected[0, :], projected[1, :], 'r-o')
    plt.title('corrected points')
    #plt.savefig('homographyCorrected.eps', format='eps', dpi=1000)

    plt.show()
    
###################  main  ###################
if __name__ == "__main__" :
    #
    #  DO NOT MODIFY THIS SECTION
    #
    
    
    # points in skewed picture to use for mapping
    # each column is a vector representing the x/y coordinates of a point
    #   in the skewed picture
    # row 1 is the x-values
    # row 2 is the y-values
    points_skewed = np.array([[0,   0, 50, 50],
                              [0, 100, 75, 25]])
    
    # points in corrected picture to use for mapping
    # each column is a vector representing the x/y coordinates of a point
    #   in the corrected picture
    # row 1 is the x-values
    # row 2 is the y-values
    points_corrected = np.array([[0,   0, 200, 200], 
                                 [0, 100, 100,   0 ]])
    
    # each column is the set of x/y coordinates of a point that should be corrected
    test_points = np.array([[42.8571, 38.4615, 33.3333, 27.2727, 20, 11.1111, 20, \
                             27.2727, 33.3333, 38.4615, 42.8571, 46.6667, 42.8571],
                            [50,      65.3846, 66.6667, 68.1818, 50, 27.7778, 30, \
                             31.8182, 33.3333, 34.6154, 35.7143, 36.6667, 50],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
      
    H, projected = processPoints(points_skewed, points_corrected, test_points)
    print('part a: H matrix\n')
    print(H)
    print('\npart b: projected test points (not as homogeneous coordinates)\n')
    print(projected)
    
    plotPoints(projected, test_points) 
