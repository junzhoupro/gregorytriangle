import numpy as np
import sympy as sp
from scipy.optimize import least_squares
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

# ~ print( G.subs([(u, 0.5), (v, 0.25)]) )
#evalG = ndGdu(0.5, 0.25)

# ~ print(evalG) # This should be a bit faster than G.subs(...)
#print(evalG.shape)
#quit()

# ---

# Use 2-fold symmetry here
def evalFuncs(uVals, vVals, indsRowCol):
    u, v = uVals, vVals
    F = ndGdu

    numOfVals = len(uVals)
    evalsF = np.zeros([2, 15, numOfVals])
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F(u[k], v[k])[j]
        evalsF[0][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F(1-u[k]-v[k], v[k])[j]
        evalsF[1][j] = row
    funcProducts = evalsF[:, indsRowCol, :]
    return np.sum(funcProducts, axis=0)

# Use single median node here
def evalFuncsSingle(uVal, vVal, indsRowCol):
    u, v = uVal, vVal
    F = ndGdu
    #print("u", u)
    #evalF = np.array([ F(u, u) ])[:,:,0,:]
    numOfVals = len(u)
    evalsF = np.zeros([1, 15, numOfVals])
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            #row[k] = F(u[k], (1-2*u[k]))[j]
            row[k] = F(u[k], 1-u[k]-u[k])[j]
        evalsF[0][j] = row
    funcProducts = evalsF[:, indsRowCol, :]
    #print("evalsF")
    #print(evalsF)
    #print("funcProducts+++++++++")
    #print(funcProducts)
    #print("np.sum(funcProducts, axis=0)+++++++++")
    #print(np.sum(funcProducts, axis=0))
    return np.sum(funcProducts, axis=0)

# ---

# optVars are the Pts and Wts
# exactInts contains the array of exact integrals
def currentResiduals(optVars, exactInts, indsRowCol):
    #  u u ... v v ... w w....
    numSinglePt = NUMOFSINGLEPOINT #
    numTwoSymPt = NUMOFTWOSYMPOINT #
    numTotal = numSinglePt + numTwoSymPt

    us = optVars[:numTwoSymPt]
    vs = optVars[numTotal:numTotal+numTwoSymPt]
    ws = optVars[2*numTotal:(2*numTotal+numTwoSymPt)]
	
    singleU = optVars[numTwoSymPt:numTotal]
    singleV = optVars[numTotal+numTwoSymPt:2*numTotal]
    singleW = optVars[2*numTotal+numTwoSymPt:]

	#print("all")
	#print("us:", us)
	#print("vs:", vs)
	#print("ws:", ws)
	#print("singleU:", singleU)
	#print("singleV:", singleV)
	#print("singleW:", singleW)
	#quit()
	
    funcEvals = evalFuncs(us, vs, indsRowCol)
    funcEvalSingle = evalFuncsSingle(singleU, singleV, indsRowCol)
	#currentWeight = np.array([optVars[1]])
	#centreWeight = np.array([optVars[2]])
	#print("funcEvals====")
	#print(funcEvals)
	#print("currentWeight")
	#print(currentWeight)
	#print("centreWeight")
	#print(centreWeight)
	#print("funcEvals.dot(currentWeight)")
	#print(funcEvals.dot(currentWeight))
	#print("funcEvals.dot(currentWeight) + funcEvalCentre.dot(centreWeight)")
	#print(funcEvals.dot(currentWeight) + funcEvalCentre.dot(centreWeight))
	#print("funcEvalCentre.dot(centreWeight)")
	#print(funcEvalCentre.dot(centreWeight))
	#print("+++++++++++++++funcEvals.dot(currentWeight) + funcEvalCentre.dot(centreWeight) - exactInts")
	#print(funcEvals.dot(currentWeight) + funcEvalCentre.dot(centreWeight) - exactInts)
	#print("22222:", funcEvals.dot(ws))
	#print("11111:", funcEvalSingle.dot(singleW))


    return funcEvals.dot(ws) + funcEvalSingle.dot(singleW) - exactInts

# ---


# ---
# The Python function determineSymmetries is a somewhat ad-hoc function I wrote to (numerically)
# figure out which products (of the 15*15 = 225 resulting mathematical functions) can be omitted due to symmetry.
# There might very well be a better/more elegant way to do this than relying on some random samples rounded to a certain number of digits ;).
def determineSymmetries():

    u3, v3 = 0.27, 0.53
    P = np.array([[u3, v3], [u3, 1-u3-v3], [v3, 1-u3-v3], [1-u3-v3, v3], [1-u3-v3, u3], [v3, u3]])
    print(P)
    #R = ndGdu(P[:,0], P[:,1])
    #print("RRRRRR", R)
    R = np.zeros([15, 1, 6])
    for i in range(15):
        row = np.zeros([1, 6])
        row[0][0] = ndGdu(P[0,0], P[0,1])[i]
        row[0][1] = ndGdu(P[1,0], P[1,1])[i]
        row[0][2] = ndGdu(P[2,0], P[2,1])[i]
        row[0][3] = ndGdu(P[3,0], P[3,1])[i]
        row[0][4] = ndGdu(P[4,0], P[4,1])[i]
        row[0][5] = ndGdu(P[5,0], P[5,1])[i]
        #print("i ", i, row)
        R[i] = row

        
    print("RRRRRR", R)

    L = np.zeros(15)

    for k in range(15):
        val = (np.abs(R[k, 0, :]).round(10))
        L[k] = np.prod(val).round(10)
        #print("L",k,L[k])
               



    U, I = np.unique(L, return_index = True)
    print(U.size)
    print("U")
    print(U)
    print("I")
    print(I)
    #print(U.size)
    # ~ print(L.flatten()[I]) # Flatten handles entries row by row
    indexSort = np.sort(I)
    print("indexSort")
    print(indexSort)
    return indexSort

    indsRowCol = np.array([ I // 15, I % 15 ])
    print("indsRowCol")
    print(indsRowCol)

	# For testing whether the obtained quadratures work for all 400 products
	# ~ k = 20
	# ~ indsRowCol = np.zeros((2, k*k), dtype=np.int64)
	# ~ indsRowCol[0, :] = np.repeat(np.arange(0, k), k)
	# ~ indsRowCol[1, :] = np.tile(np.arange(0, k), k)

    lexSort = np.lexsort(( indsRowCol[1, :], indsRowCol[0, :] )) # Sorted based on LAST array of coefficients
    print("lexSort")
    print(lexSort)
    print("indsRowCol[:, lexSort]")
    print(indsRowCol[:, lexSort])

    # ~ return np.array([[2, 9, 14], [0, 0, 0]]) # Only for testing with G
    #print("groupedL")
    #print(groupedL)
    return indsRowCol[:, lexSort]


def computeExactInts(indsRowCol):
	# Compute exact integrals... Rather slow! Perhaps use Maple for this?
	#print("computeExactInts")
	#print(indsRowCol.shape[1])
    for k in range(len(indsRowCol)):
        F = dGdu[indsRowCol[k]].simplify()
		#F = (dGdu[indsRowCol[0, k]] * dGdv[indsRowCol[1, k]]).simplify()
		# ~ F = dGdu[indsRowCol[0, k]] * dGdv[indsRowCol[1, k]]
		#print("k:.............", k)
        #print(F, ",", sep = '')
        #quit()

    return np.array([0.250000000000000000000000, 0., 0., 0.250000000000000000000000, 0.250000000000000000000000, 0., 0., 0.])
    #return np.array([], dtype=np.longdouble)

# ---

def runOptimisation():

    print("determineSymmetries")
    indsRowCol = determineSymmetries()
    print(indsRowCol)

    exactInts = computeExactInts(indsRowCol)
    print("exactInts")
    print(exactInts)
    print("Unique integrals:", len(np.unique(exactInts)))

    #single point on median line
    numSinglePt = NUMOFSINGLEPOINT #
    #2-fold symmetric points 
    numTwoSymPt = NUMOFTWOSYMPOINT #
    numTotal = numSinglePt + numTwoSymPt
    #initGuess = np.empty(3*numPtsWts, dtype=np.longdouble)
    initGuess = np.empty(3*numTwoSymPt+3*numSinglePt, dtype=np.longdouble)
    print("initGuess")
    print(initGuess)
    #  u u... v v ... w w....

    # Bounds on Pts and Wts
    Eps = 1e-3
    lowerBoundsPts = Eps + np.zeros(2*(numSinglePt+numTwoSymPt), dtype=np.longdouble) # Careful with singularities at corners of the domain!
    lowerBoundsWts = np.ones((numSinglePt+numTwoSymPt), dtype=np.longdouble) # -np.inf * np.ones(...)
    lowerBounds = np.concatenate((lowerBoundsPts, lowerBoundsWts.dot(-1)))
    print("lowerBounds")
    print(lowerBounds)

    upperBoundsPts = np.ones(2*(numSinglePt+numTwoSymPt), dtype=np.longdouble) - Eps # Careful with singularities at corners of the domain!
    upperBoundsWts = np.ones((numSinglePt+numTwoSymPt), dtype=np.longdouble) # np.inf * np.ones(...)
    upperBounds = np.concatenate((upperBoundsPts, upperBoundsWts))
    print("upperBounds")
    print(upperBounds)

    numConverged = 0
    maxConverged = 1 #5

    resTol = 2.24e-16 #2.24e-16
    varTol = 1e-18 #1e-18

    maxEvals = 5e3 #1e4 #3e4 #5e3 # To be tweaked
    markers = ['o', 'x', '*', 'D', 's']

    while (numConverged < maxConverged):
        

		# Set initial guess
        for k in range(numTwoSymPt+numSinglePt):
		    # Point (u,v)... Take into account the bounds!
            initGuess[k*3] = Eps + (1 - 2*Eps) * np.random.rand() #
		    #initGuess[numOnVertLine + k] = Eps + (1 - 2*Eps) * np.random.rand() #
		    # Weight w
		    #initGuess[2*numPtsWts + k] = np.random.rand() / 8 # Or / 8
            initGuess[k*3 + 1] = Eps + (1 - 2*Eps) * np.random.rand()
            initGuess[k*3 + 2] = Eps + (1 - 2*Eps) * np.random.rand()

        print("initGuess====",initGuess)

		# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
		# Methods are 'trf', 'dogbox' and 'lm'
        leastSq = least_squares(currentResiduals, initGuess, jac='2-point',
                                    bounds=(lowerBounds, upperBounds), method='trf',
                                    ftol=resTol, xtol=varTol, gtol=None, max_nfev=maxEvals,
                                    args=(exactInts, indsRowCol), verbose = 1) # 2 # jac=currentJacobian

        if leastSq.success:
            currentSol = leastSq.x
            print("currentSol")
            print(currentSol)
            relErrors = currentResiduals(currentSol, exactInts, indsRowCol) #if (0 in exactInts) else currentResiduals(currentSol, exactInts, indsRowCol) / exactInts

            if np.max(np.abs(relErrors)) < 1e-16:
                print(":: [", numConverged+1, "/", maxConverged, "] Converged after ", leastSq.nfev, " iterations", sep = '')
                print(relErrors)

			    #PtsU = currentSol[:numPtsWts]
			    #PtsV = currentSol[numPtsWts:2*numPtsWts]
			    #Wts = currentSol[2*numPtsWts:]
			    #PtsU = currentSol[:1]
			    #PtsV = currentSol[:1]
			    #Wts = currentSol[1:]
                us = currentSol[:numTwoSymPt]
                vs = currentSol[numTotal:numTotal+numTwoSymPt]
                ws = currentSol[2*numTotal:(2*numTotal+numTwoSymPt)]
                singleU = currentSol[numTwoSymPt:numTotal]
                singleV = currentSol[numTotal+numTwoSymPt:2*numTotal]
                singleW = currentSol[2*numTotal+numTwoSymPt:]

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
                print([us, 1-us-vs, singleU])
			    #print([1-us-us, us, 1-singleU-singleU])
                print([vs, vs, 1-singleU-singleU])
                print("WEIGHTs:")
                print([ws, ws, singleW])
                uAll = np.append(np.append(us, 1-us-vs),singleU)
			    #vAll = np.append(np.append(1-us-us, us), 1-singleU-singleU)
                vAll = np.append(np.append(vs, vs), 1-singleU-singleU)
                wAll = np.append(np.append(ws, ws), singleW)
                print("uAll",uAll)
                print("vAll",vAll)
                print("wAll",wAll)

                pointsAll = []
                for point in range(NUMOFTWOSYMPOINT * 2 + NUMOFSINGLEPOINT):
                    pointsAll.append([uAll[point], vAll[point], 1-uAll[point]-vAll[point]])

                flag_negative = 0
                for x in range(NUMOFTWOSYMPOINT):
                    if(us[x] > 0 and 1-us[x]-vs[x] > 0 and us[x] < 0.9 and 1-us[x]-vs[x] < 0.9):
                        print("U2 OK")
                    else:
                        flag_negative = 1
                for y in range(NUMOFSINGLEPOINT):
                    if(singleU[y] > 0 and singleU[y] < 0.9):
                        print("U1 OK")
                    else:
                        flag_negative = 1
                        
                for x in range(NUMOFTWOSYMPOINT):
                    if(vs[x] > 0.1 and vs[x] < 0.9):
                        print("V2 OK")
                    else:
                        flag_negative = 1
                for y in range(NUMOFSINGLEPOINT):
                    if(1-singleU[y]-singleU[y] > 0 and 1-singleU[y]-singleU[y] < 0.9):
                        print("V1 OK")
                    else:
                        flag_negative = 1

                for x in range(NUMOFTWOSYMPOINT):
                    if(ws[x] > 0):
                        print("W2 OK")
                    else:
                        flag_negative = 1
                for y in range(NUMOFSINGLEPOINT):
                    if(singleW[y] > 0):
                        print("W1 OK")
                    else:
                        flag_negative = 1

                if (flag_negative == 1):
                    print("Continue")
                    continue

                f = open("Gu.txt", "a")
                f.write("U:\n")
                f.write("[")
                
                for x in range(NUMOFTWOSYMPOINT):
                    f.write(str(us[x]) + ",")
                    f.write(str(1-us[x]-vs[x]) + ",")
                for y in range(NUMOFSINGLEPOINT):
                    if(y+1 == NUMOFSINGLEPOINT):
                        f.write(str(singleU[y]))
                    else:
                        f.write(str(singleU[y]) + ",")
                    f.write("]\n")


                f.write("V:\n")
                f.write("[")
                for x in range(NUMOFTWOSYMPOINT):
                    f.write(str(vs[x]) + ",")
                    f.write(str(vs[x]) + ",")
                for y in range(NUMOFSINGLEPOINT):
                    if(y+1 == NUMOFSINGLEPOINT):
                        f.write(str(1-singleU[y]-singleU[y]))
                    else:
                        f.write(str(1-singleU[y]-singleU[y]) + ",")
                    f.write("]\n")


                f.write("W:\n")
                f.write("[")
                for x in range(NUMOFTWOSYMPOINT):
                    f.write(str(ws[x]) + ",")
                    f.write(str(ws[x]) + ",")
                for y in range(NUMOFSINGLEPOINT):
                    if(y+1 == NUMOFSINGLEPOINT):
                        f.write(str(singleW[y]))
                    else:
                        f.write(str(singleW[y]) + ",")
                    f.write("]\n\n")

				# Plot stuff
##			    plt.scatter([PtsU, 1-PtsU, 1-PtsV, 1-PtsV, 1-PtsU, PtsU, PtsV, PtsV], [PtsV, PtsV, PtsU, 1-PtsU, 1-PtsV, 1-PtsV, 1-PtsU, PtsU], s=1e4*np.abs([Wts, Wts, Wts, Wts, Wts, Wts, Wts, Wts]), color=plt.cm.hsv(numConverged/maxConverged), edgecolors='none', alpha=0.5, label=str(numConverged)) # 0.2 # 0.01

                
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
                            #color=plt.cm.hsv(numConverged/maxConverged),
                            color='black',
                            edgecolors='none', alpha=1,
                            label=str(numConverged))
			    #color='tab:blue'

                numConverged += 1

            else:
                print(".. Successful, but minimal (absolute) error not low enough (", np.max(np.abs(relErrors)), ")", sep = '')

        else:
            print("Not successful...")

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

    plt.gcf().canvas.manager.set_window_title("Gregory product quadrature experiments, numPtsWts = " + str(numTotal))
    plt.show()

    


# ---

if __name__ == "__main__":
    runOptimisation()
