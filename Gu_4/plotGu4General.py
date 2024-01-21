import numpy as np
import sympy as sp
from scipy.optimize import least_squares
#Python 3.8.8

# ~ import matplotlib
# ~ matplotlib.use('TkAgg') # Qt5Agg, TkAgg
import matplotlib.pyplot as plt
import math


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

NUMOFSINGLEPOINT = 4 #

# 3 points v0, v1, v2， counterclockwise
# v0
vertices = np.array([[-1, 0], [1, 0], [0, pow(3, 0.5)]])
print("vertices")
print(vertices.sum(axis=0))
print(vertices.sum(axis=0)[0])
print(vertices.sum(axis=0)[1])


def runOptimisation():

    markers = ['o', 'x', '*', 'D', 's']

    us = np.empty(4, dtype=np.longdouble)
    #us[0] = 0.045647706332150193450
    #us[1] = 0.075473742212925014045
    #us[0] = 0.103626225586000556461904597516
    #us[1] = 0.000844678053576423380537545478370

    us[0] = 0.2912795938596667
    us[1] = 0.01825632022287768
    us[2] = 0.8487798289638642
    us[3] = 0.2808718208401459
                #us = currentSol[:numTwoSymPt] np.zeros[]
    vs = np.empty(4, dtype=np.longdouble)
    #vs[0] = 0.78399273438728457282
    #vs[1] = 0.21005718414920775927
    vs[0] = 0.5730624372335545
    vs[1] = 0.5730624372335545
    vs[2] = 0.10159027564152055
    vs[3] = 0.10159027564152055
                #vs = currentSol[numTotal:numTotal+numTwoSymPt]
    ws = np.empty(4, dtype=np.longdouble)
    #ws[0] = 0.05369773188513451277
    #ws[1] = 0.10000000000000000000
    ws[0] = 0.1757229760308316
    ws[1] = 0.07004239101573925
    ws[2] = 0.07535743177873247
    ws[3] = 0.17887720117469671
    
                #ws = currentSol[2*numTotal:(2*numTotal+numTwoSymPt)]
                

			    # Sort
			    #PtsU = PtsU[Wts.argsort()]
			    #PtsV = PtsV[Wts.argsort()]
			    #Wts.sort()

			    #print("-> PtsU =", list(PtsU)) # List, to add separating commas between entries
			    #print("-> Wts =", list(PtsV))
			    #print("-> Wts =", list(Wts))
			    #print("-> Sum(Wts) =", np.sum(Wts))
    print("")
    print("POINTs:")
			    #print([us, us, singleV])
			    #print([1-us-us, us, 1-singleU-singleU])
			    #print([us, 1-us-us, 1-singleV-singleV])
    print([us, vs])
			    #print([1-us-us, us, 1-singleU-singleU])
    print([1-us-vs, vs])
    print("WEIGHTs:")
    print([ws, ws])
    uAll = us
			    #vAll = np.append(np.append(1-us-us, us), 1-singleU-singleU)
    vAll = vs
    wAll = ws
    print("uAll",uAll)
    print("vAll",vAll)
    print("wAll",wAll)

    pointsAll = []
    for point in range(NUMOFSINGLEPOINT):
        pointsAll.append([uAll[point], vAll[point], 1-uAll[point]-vAll[point]])
    print("pointsAll")
    print(pointsAll)

    uAll2 = 1-us-vs
    vAll2 = vs
    wAll2 = ws

    pointsAll2 = []
    for point in range(NUMOFSINGLEPOINT):
        pointsAll2.append([uAll2[point], vAll2[point], 1-uAll2[point]-vAll2[point]])
    print("pointsAll2")
    print(pointsAll2)


                
    plt.scatter([np.dot(pointsAll[0], vertices[:,0]),
                 np.dot(pointsAll[1], vertices[:,0]),
                 np.dot(pointsAll[2], vertices[:,0]),
                 np.dot(pointsAll[3], vertices[:,0])],
                [np.dot(pointsAll[0], vertices[:,1]),
                 np.dot(pointsAll[1], vertices[:,1]),
                 np.dot(pointsAll[2], vertices[:,1]),
                 np.dot(pointsAll[3], vertices[:,1])],
                s=1e3*np.abs([wAll]),
                marker=markers[0],
                #color=plt.cm.hsv(0),
                color='black',
                edgecolors='none', alpha=1,
                label=str(0))
			    #color='tab:blue'
##    plt.scatter([np.dot(pointsAll2[0], vertices[:,0]),
##                 np.dot(pointsAll2[1], vertices[:,0]),
##                 np.dot(pointsAll2[2], vertices[:,0]),
##                 np.dot(pointsAll2[3], vertices[:,0])],
##                 [np.dot(pointsAll2[0], vertices[:,1]),
##                 np.dot(pointsAll2[1], vertices[:,1]),
##                 np.dot(pointsAll2[2], vertices[:,1]),
##                 np.dot(pointsAll2[3], vertices[:,1])],
##                 s=1e3*np.abs([wAll2]),
##                 marker=markers[0],
##                 color=plt.cm.hsv(0.5),
##                            #color='black',
##                 edgecolors='none', alpha=1,
##                 label=str(0))


    # Plot commands
    plt.axis('square')
    plt.gca().set(xlim=(-1, 1))
    plt.gca().set(ylim=(-0.1, pow(3, 0.5)))
    #plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) # ["0", "1", ...]
    plt.axis('off')
    ax = plt.gca()
    #plt.gca().spines['right'].set_visible(False)
    #plt.gca().spines['top'].set_visible(False)
    #plt.gca().spines['left'].set_linewidth('0.7')
    #plt.gca().spines['bottom'].set_linewidth('0.7')
    plt.plot([-1, 0], [0, pow(3, 0.5)], color='black', linewidth=0.7)
    plt.plot([1, 0], [0, pow(3, 0.5)], color='black', linewidth=0.7)
    plt.plot([1, -1], [0, 0], color='black', linewidth=0.7)

    plt.subplots_adjust(right=0.98, left=0.02, top=0.98, bottom=0.01)
    plt.show()

    


# ---

if __name__ == "__main__":
    runOptimisation()
