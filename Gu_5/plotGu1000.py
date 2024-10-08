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
#  9  8  7  6
# 10        5
# 11        4
#  0  1  2  3

p0,p1,p2,e00,e01,e10,e11,e20,e21,f00,f01,f10,f11,f20,f21 = sp.symbols('p0,p1,p2,e00,e01,e10,e11,e20,e21,f00,f01,f10,f11,f20,f21')
cps = np.array([p0,p1,p2,e00,e01,e10,e11,e20,e21,f00,f01,f10,f11,f20,f21])

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
print(sp.pretty(dGdu))

# ---

nG = sp.lambdify((u, v), G, "numpy") # [u, v]
ndGdu = sp.lambdify((u, v), dGdu, "numpy")
ndGdv = sp.lambdify((u, v), dGdv, "numpy")
nddGdudv = sp.lambdify((u, v), ddGdudv, "numpy")


# ~ print(dGdu)
print(nG, ndGdu)

NUMOFSINGLEPOINT = 1 #
NUMOFTWOSYMPOINT = 2 #

# 3 points v0, v1, v2， counterclockwise
# v0
vertices = np.array([[-1, 0], [1, 0], [0, pow(3, 0.5)]])
print("vertices")
print(vertices.sum(axis=0))
print(vertices.sum(axis=0)[0])
print(vertices.sum(axis=0)[1])

#read data from xls
#data = pd.read_excel('Gu1000.xls')
data = pd.read_excel('Gu10003.xls',sheet_name='play')
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
print("3,3:" + str(data.iloc[3,3]))
print("0,3:" + str(data.iloc[0,3]))
print("1,0:" + str(data.iloc[1,0]))

def runOptimisation(iRow):

    markers = ['o', 'x', '*', 'D', 's']

    us = np.empty(2, dtype=np.longdouble)
    us[0] = data.iloc[iRow,0]
    us[1] = data.iloc[iRow,1]
                #us = currentSol[:numTwoSymPt] np.zeros[]
    vs = np.empty(2, dtype=np.longdouble)
    vs[0] = data.iloc[iRow,2]
    vs[1] = data.iloc[iRow,3]
                #vs = currentSol[numTotal:numTotal+numTwoSymPt]
    ws = np.empty(2, dtype=np.longdouble)
    ws[0] = data.iloc[iRow,4]
    ws[1] = data.iloc[iRow,5]
                #ws = currentSol[2*numTotal:(2*numTotal+numTwoSymPt)]
    singleU = np.empty(1, dtype=np.longdouble)
    singleV = np.empty(1, dtype=np.longdouble)
    singleW = np.empty(1, dtype=np.longdouble)

    singleU[0] = 1/2 - (vs[1])/2
    singleV[0] = vs[1]
    singleW[0] = 1/2-2*(ws[0] + ws[1])
                

    #print("")
    #print("POINTs:")

    #print([us, 1-us-vs, singleU])
			    
    #print([vs, vs, 1-singleU-singleU])
    #print("WEIGHTs:")
    #print([ws, ws, singleW])
    uAll = np.append(np.append(us, 1-us-vs),singleU)
			   
    vAll = np.append(np.append(vs, vs), 1-singleU-singleU)
    wAll = np.append(np.append(ws, ws), singleW)
    #print("uAll",uAll)
    #print("vAll",vAll)
    #print("wAll",wAll)

    pointsAll = []
    for point in range(NUMOFTWOSYMPOINT * 2 + NUMOFSINGLEPOINT):
        pointsAll.append([uAll[point], vAll[point], 1-uAll[point]-vAll[point]])
    #print("pointsAll")
    #print(pointsAll)


                
    plt.scatter([np.dot(pointsAll[0], vertices[:,0]),
                 np.dot(pointsAll[1], vertices[:,0]),
                 np.dot(pointsAll[2], vertices[:,0]),
                 np.dot(pointsAll[3], vertices[:,0]),
                 np.dot(pointsAll[4], vertices[:,0])],
                [np.dot(pointsAll[0], vertices[:,1]),
                 np.dot(pointsAll[1], vertices[:,1]),
                 np.dot(pointsAll[2], vertices[:,1]),
                 np.dot(pointsAll[3], vertices[:,1]),
                 np.dot(pointsAll[4], vertices[:,1])],
                s=1e3*np.abs([wAll]),
                marker=markers[0],
                color=plt.cm.hsv(iRow/nrows),
                #color='black',
                edgecolors='none', alpha=0.1,
                label=str(0))
			    #color='tab:blue'




# ---

if __name__ == "__main__":
    for iRow in range(nrows):
        runOptimisation(iRow)

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
