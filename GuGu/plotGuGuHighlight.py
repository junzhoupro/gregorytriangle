import numpy as np
import sympy as sp
from scipy.optimize import least_squares
import pandas as pd
#Python 3.8.8

# ~ import matplotlib
# ~ matplotlib.use('TkAgg') # Qt5Agg, TkAgg
import matplotlib.pyplot as plt


np.set_printoptions(precision = 18)
np.set_printoptions(linewidth = 180)

u, v, w = sp.symbols('u v w')
mu, mv, w = 1-u, 1-v, 1-u-v
u1, u2, u3, v1, v2, v3 = sp.symbols('u1, u2, u3, v1, v2, v3')

p0,p1,p2,e00,e01,e10,e11,e20,e21,f00,f01,f10,f11,f20,f21 = sp.symbols('p0,p1,p2,e00,e01,e10,e11,e20,e21,f00,f01,f10,f11,f20,f21')
cps = np.array([p0,p1,p2,e00,e01,e10,e11,e20,e21,f00,f01,f10,f11,f20,f21])

# indices are like:
# 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
# p0,p1,p2,e00,e01,e10,e11,e20,e21,f00,f01,f10,f11,f20,f21

F0 = (w*f00+v*f01)/(v+w)
F1 = (u*f10+w*f11)/(w+u)
F2 = (v*f20+u*f21)/(u+v)                 
T = (u)**(3)*p0+(v)**(3)*p1+(w)**(3)*p2 + 3*u*v*(u+v)*(u*e01+v*e10) + 3*v*w*(v+w)*(v*e11+w*e20) + 3*w*u*(w+u)*(w*e21+u*e00) + 12*u*v*w*(u*F0+v*F1+w*F2)

G = sp.Matrix()
for cp in cps:
    print(cp)
    G = sp.Matrix([G, sp.factor(sp.factor(T.expand().coeff(cp)))])

#print(sp.pretty(G))
print("G:")
print(sp.pretty(G))
print(G.shape)


# ---

print("Computing symbolic derivatives of G...")
dGdu = sp.diff(G, u) #.simplify()
#print(sp.pretty(dGdu))
#print(dGdu.shape)
dGdv = sp.diff(G, v) #.simplify()
ddGdudv = sp.diff(G, u, v) #.simplify()
print(sp.pretty(ddGdudv))

# ---

nG = sp.lambdify((u, v), G, "numpy") # [u, v]
ndGdu = sp.lambdify((u, v), dGdu, "numpy")
ndGdv = sp.lambdify((u, v), dGdv, "numpy")
nddGdudv = sp.lambdify((u, v), ddGdudv, "numpy")


# ~ print(dGdu)
print(nG, nddGdudv)

# Set the number of single point
NUMOFSINGLEPOINT = 0 #
# Set the number of 2-fold symmetric points
NUMOFTWOSYMPOINT = 9 #

# 3 points v0, v1, v2， counterclockwise
# v0
vertices = np.array([[-1, 0], [1, 0], [0, pow(3, 0.5)]])
print("vertices")
print(vertices.sum(axis=0))
print(vertices.sum(axis=0)[0])
print(vertices.sum(axis=0)[1])

#read data from xls
#data = pd.read_excel('Guv1000.xls')
data = pd.read_excel('GuGuSymmetric631.xls',sheet_name='play')
print(data.shape)
print(data[0:3])
print("DATA")
print(data)
print(data.columns)
print(data.index)
nrows = data.shape[0]
ncols = data.columns.size
print("=========================================================================")
print('Max Rows:'+str(nrows))
print('Max Columns:'+str(ncols))
#print("3,3:" + str(data.iloc[3,3]))
#print("0,3:" + str(data.iloc[0,3]))
#print("1,0:" + str(data.iloc[1,0]))

def runOptimisation(iRow):

    markers = ['o', 'x', '*', 'D', 's']


    us2 = np.empty(18, dtype=np.longdouble)
    for i in range(18):
        us2[i] = data.iloc[iRow,i]

        
                #us = currentSol[:numTwoSymPt] np.zeros[]
    vs2 = np.empty(18, dtype=np.longdouble)
    for i in range(18):
        vs2[i] = data.iloc[iRow,i+18]

    
                #vs = currentSol[numTotal:numTotal+numTwoSymPt]
    ws2 = np.empty(18, dtype=np.longdouble)
    for i in range(18):
        ws2[i] = data.iloc[iRow,i+18+18]


                #ws = currentSol[2*numTotal:(2*numTotal+numTwoSymPt)]
    
                
    #print("")
    #print("POINTs:")
    #print([us, vs, singleU])
    #print([vs, us, singleU])
    #print("WEIGHTs:")
    #print([ws, ws, singleW])
    uAll = np.array(us2)

    vAll = np.array(vs2)
    wAll = np.array(ws2)
    print("uAll",uAll)
    print("vAll",vAll)
    print("wAll",wAll)

    pointsAll = []
    for point in range(NUMOFTWOSYMPOINT * 2):
        pointsAll.append([uAll[point], vAll[point], 1-uAll[point]-vAll[point]])
    print("pointsAll:")
    print(pointsAll)

    pointsX = []
    pointsY = []
    for point in range(NUMOFTWOSYMPOINT * 2):
        pointsX.append(np.dot(pointsAll[point], vertices[:,0]))
        pointsY.append(np.dot(pointsAll[point], vertices[:,1]))


    plt.scatter(pointsX, pointsY,
                s=1e4*np.abs([wAll/4]),
                marker=markers[0],
                #color=plt.cm.hsv(numConverged/maxConverged),
                color='black',
                edgecolors='none', alpha=0.3,
                label=str(3))

    pointsU = []
    pointsV = []
    pointsW = np.empty(NUMOFTWOSYMPOINT, dtype=np.longdouble)
    
    pointsW[2] = wAll[13]
    for point in range(NUMOFTWOSYMPOINT):
        if uAll[point] > uAll[point + NUMOFTWOSYMPOINT]:
            pointsU.append(np.dot(pointsAll[point], vertices[:,0]))
            pointsV.append(np.dot(pointsAll[point], vertices[:,1]))
            pointsW[point] = wAll[point]
        else:
            pointsU.append(np.dot(pointsAll[point + NUMOFTWOSYMPOINT], vertices[:,0]))
            pointsV.append(np.dot(pointsAll[point + NUMOFTWOSYMPOINT], vertices[:,1]))
            pointsW[point] = wAll[point + NUMOFTWOSYMPOINT]

    print("pointsu")
    print(pointsU)
    print("pointsv")
    print(pointsV)
    print("pointsw")
    print(pointsW)

    plt.scatter(pointsU, pointsV,
                s=1e4*np.abs([pointsW/4]),
                marker=markers[0],
                #color=plt.cm.hsv(numConverged/maxConverged),
                #color=plt.cm.hsv(iRow/3),
                color='black',
                edgecolors='none', alpha=1,
                label=str(3))



# ---

if __name__ == "__main__":
    runOptimisation(1)
    #for iRow in range(nrows):
     #   runOptimisation(iRow)

    # Plot commands
    plt.axis('square')
    plt.gca().set(xlim=(-1, 1))
    plt.gca().set(ylim=(-0.1, pow(3, 0.5)))
    #plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) # ["0", "1", ...]
    plt.axis('off')
    
    #plt.gca().spines['right'].set_visible(False)
    #plt.gca().spines['top'].set_visible(False)
    #plt.gca().spines['left'].set_linewidth('0.7')
    #plt.gca().spines['bottom'].set_linewidth('0.7')
    plt.plot([-1, 0], [0, pow(3, 0.5)], color='black', linewidth=0.7)
    plt.plot([1, 0], [0, pow(3, 0.5)], color='black', linewidth=0.7)
    plt.plot([1, -1], [0, 0], color='black', linewidth=0.7)

    plt.subplots_adjust(right=0.98, left=0.02, top=0.98, bottom=0.01)
    plt.show()
