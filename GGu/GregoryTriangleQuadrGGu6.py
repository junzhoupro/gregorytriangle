import numpy as np
import sympy as sp
from scipy.optimize import least_squares
#Python 3.8.8

# ~ import matplotlib
# ~ matplotlib.use('TkAgg') # Qt5Agg, TkAgg
import matplotlib.pyplot as plt
import xlwt,xlrd
from xlutils.copy import copy


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
print(sp.pretty(dGdu))
print(dGdu.shape)
dGdv = sp.diff(G, v) #.simplify()
# ~ ddGdudv = sp.diff(G, u, v) #.simplify()

# ---

nG = sp.lambdify((u, v), G, "numpy") # [u, v]
ndGdu = sp.lambdify((u, v), dGdu, "numpy")
ndGdv = sp.lambdify((u, v), dGdv, "numpy")

# ~ print(dGdu)
print(nG, ndGdu)

# ~ print( G.subs([(u, 0.5), (v, 0.25)]) )
evalG = ndGdu(0.5, 0.25)

# ~ print(evalG) # This should be a bit faster than G.subs(...)
print(evalG.shape)

# Set the number of single point
NUMOFSINGLEPOINT = 0 #7
# Set the number of 2-fold symmetric points
NUMOFTWOSYMPOINT = 0 #30

# Set the number of 3-fold symmetric points
NUMOFTHREESYMPOINT = 2 #
# Set the number of 6-fold symmetric points
NUMOFSIXSYMPOINT = 4 #

# 3 points v0, v1, v2， counterclockwise
# v0
vertices = np.array([[-1, 0], [1, 0], [0, pow(3, 0.5)]])
print("vertices")
print(vertices.sum(axis=0))
print(vertices.sum(axis=0)[0])
print(vertices.sum(axis=0)[1])



# This function is to evaluate the median point
def evalFuncsSingle(uVals, vVals, indsRowCol):
    u, v = uVals, vVals
    F = nG
    H = ndGdu
    numOfVals = len(uVals)

    evalsF = np.zeros([1, 15, numOfVals])
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F(1/3, 1/3)[j]
        evalsF[0][j] = row
##    for j in range(15):
##        row = np.zeros(numOfVals)
##        for k in range(numOfVals):
##            row[k] = F(u[k], 1-u[k]-u[k])[j]
##        evalsF[1][j] = row

    evalsH = np.zeros([1, 15, numOfVals])
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = H(1/3, 1/3)[j]
        evalsH[0][j] = row
##    for j in range(15):
##        row = np.zeros(numOfVals)
##        for k in range(numOfVals):
##            row[k] = F(u[k], 1-u[k]-u[k])[j]
##        evalsH[1][j] = row
            
	#evalsF = np.array([ F(u, v), F(u, 1-u-v), F(v, 1-u-v), F(1-u-v, v), F(1-u-v, u), F(v, u) ])[:,:,0,:]
	
	#print("evalsF===", evalsF)
	#evalsH = np.array([ H(u, v), H(1-u, v), H(1-v, u), H(1-v, 1-u), H(1-u, 1-v), H(u, 1-v), H(v, 1-u), H(v, u) ])[:,:,0,:]

	#funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsF[:, indsRowCol[1, :], :] # Now (4 or 8, 33, numPtsWts)
    funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsH[:, indsRowCol[1, :], :] # Now (4 or 8, 33, numPtsWts)
	# ~ print(funcProducts.shape). For testing with G, comment out the multiplication

	#print(funcProducts.shape)
    return np.sum(funcProducts, axis=0)

# Use 3-fold symmetry here:
# Because I saw the symmetry in the plots of maple
def evalFuncsThree(uVals, vVals, indsRowCol):
    u, v = uVals, vVals

    F = nG
    H = ndGdu
	# ~ F = ndGdu
	# ~ H = ndGdv

    numOfVals = len(uVals)

    evalsF = np.zeros([3, 15, numOfVals])
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F(u[k], u[k])[j]
        evalsF[0][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F(u[k], (1-u-u)[k])[j]
        evalsF[1][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F((1-u-u)[k], u[k])[j]
        evalsF[2][j] = row

    evalsH = np.zeros([3, 15, numOfVals])
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = H(u[k], v[k])[j]
        evalsH[0][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = H(u[k], (1-u-u)[k])[j]
        evalsH[1][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = H((1-u-u)[k], u[k])[j]
        evalsH[2][j] = row
	#evalsF = np.array([ F(u, v), F(u, 1-u-v), F(v, 1-u-v), F(1-u-v, v), F(1-u-v, u), F(v, u) ])[:,:,0,:]
	
	#print("evalsF===", evalsF)
	#evalsH = np.array([ H(u, v), H(1-u, v), H(1-v, u), H(1-v, 1-u), H(1-u, 1-v), H(u, 1-v), H(v, 1-u), H(v, u) ])[:,:,0,:]

	#funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsF[:, indsRowCol[1, :], :] # Now (4 or 8, 33, numPtsWts)
    funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsH[:, indsRowCol[1, :], :] # Now (4 or 8, 33, numPtsWts)
	# ~ print(funcProducts.shape). For testing with G, comment out the multiplication

	#print(funcProducts.shape)
    return np.sum(funcProducts, axis=0)

# Use 6-fold symmetry here:
# Because I saw the symmetry in the plots of maple
def evalFuncs(uVals, vVals, indsRowCol):
    u, v = uVals, vVals

    F = nG
    H = ndGdu
	# ~ F = ndGdu
	# ~ H = ndGdv

    numOfVals = len(uVals)

    evalsF = np.zeros([6, 15, numOfVals])
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F(u[k], v[k])[j]
        evalsF[0][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F(v[k], u[k])[j]
        evalsF[1][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F(v[k], (1-u-v)[k])[j]
        evalsF[2][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F(u[k], (1-u-v)[k])[j]
        evalsF[3][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F((1-u-v)[k], u[k])[j]
        evalsF[4][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = F((1-u-v)[k], v[k])[j]
        evalsF[5][j] = row

    evalsH = np.zeros([6, 15, numOfVals])
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = H(u[k], v[k])[j]
        evalsH[0][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = H(v[k], u[k])[j]
        evalsH[1][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = H(v[k], (1-u-v)[k])[j]
        evalsH[2][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = H(u[k], (1-u-v)[k])[j]
        evalsH[3][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = H((1-u-v)[k], u[k])[j]
        evalsH[4][j] = row
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            row[k] = H((1-u-v)[k], v[k])[j]
        evalsH[5][j] = row
            
	#evalsF = np.array([ F(u, v), F(u, 1-u-v), F(v, 1-u-v), F(1-u-v, v), F(1-u-v, u), F(v, u) ])[:,:,0,:]
	
	#print("evalsF===", evalsF)
	#evalsH = np.array([ H(u, v), H(1-u, v), H(1-v, u), H(1-v, 1-u), H(1-u, 1-v), H(u, 1-v), H(v, 1-u), H(v, u) ])[:,:,0,:]

	#funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsF[:, indsRowCol[1, :], :] # Now (4 or 8, 33, numPtsWts)
    funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsH[:, indsRowCol[1, :], :] # Now (4 or 8, 33, numPtsWts)
	# ~ print(funcProducts.shape). For testing with G, comment out the multiplication

	#print(funcProducts.shape)
    return np.sum(funcProducts, axis=0)
# ---


# This functions is to calculate residuals between the guess and the true integrals
# optVars are the Pts (for now all using 3-fold symmetry and centre one) and Wts
# exactInts contains the array of exact integrals
# numSinglePt: how many do we have a single point
# numTwoSymPt: how many pairs do we have a 2-fold symmetric points
# numThreeSymPt: how many groups do we have a 3-fold symmetric points
# numFourSymPt: how many groups do we have a 4-fold symmetric points
# numSixSymPt: how many groups do we have a 6-fold symmetric points

# us2, vs2, ws2 are variables to store (u,v,weights) of the possible 2-fold symmetric points
# us3, vs3, ws3 are variables to store (u,v,weights) of the possible 3-fold symmetric points
# us4, vs4, ws4 are variables to store (u,v,weights) of the possible 4-fold symmetric points
# us6, vs6, ws6 are variables to store (u,v,weights) of the possible 6-fold symmetric points
# singleU, singleV, singleW are variables to store (u,v,weights) of the possible single points

# funcEvals2: evaluation of the possible 2-fold symmetric points
# funcEvals3: evaluation of the possible 3-fold symmetric points
# funcEvals4: evaluation of the possible 4-fold symmetric points
# funcEvals6: evaluation of the possible 6-fold symmetric points
# funcEvalSingle: evaluation of the possible single points
def currentResiduals(optVars, exactInts, indsRowCol):

    #numPtsWts = len(optVars) // 3
    numOnVertLine = NUMOFTHREESYMPOINT #
    numSixPts = NUMOFSIXSYMPOINT #
    numSingle = NUMOFSINGLEPOINT

    #numSinglePt = NUMOFSINGLEPOINT #
    #numOnVertLine = NUMOFTHREEPOINT # u, v = v, u
    #numSixPts = NUMOFSIXPOINT
    numTotal =  numOnVertLine + numSixPts + numSingle

    us6 = optVars[:numSixPts]
    vs6 = optVars[numTotal:numTotal+numSixPts]
    ws6 = optVars[2*numTotal:(2*numTotal+numSixPts)]

    us3 = optVars[numSixPts:numSixPts+numOnVertLine]
    vs3 = optVars[numTotal+numSixPts:numTotal+numSixPts+numOnVertLine]
    ws3 = optVars[2*numTotal+numSixPts:(2*numTotal+numSixPts+numOnVertLine)]

    us1 = optVars[numSixPts+numOnVertLine:numTotal]
    vs1 = optVars[numTotal+numSixPts+numOnVertLine:numTotal*2]
    ws1 = optVars[2*numTotal+numSixPts+numOnVertLine:3*numTotal]

    funcEvals6 = evalFuncs(us6,
                           vs6,
                           indsRowCol)
    funcEvals3 = evalFuncsThree(us3,
                                vs3,
                                indsRowCol)
    funcEvals1 = evalFuncsSingle(us1,
                                vs1,
                                indsRowCol)
	
    return funcEvals6.dot(ws6) + funcEvals3.dot(ws3) + funcEvals1.dot(ws1) - exactInts# + funcEvals3.dot(ws3) + funcEvals4.dot(ws4) + funcEvals6.dot(ws6) + funcEvalSingle.dot(singleW) - exactInts

# ---

# ---
# The Python function determineSymmetries is a somewhat ad-hoc function I wrote to (numerically)
# figure out which products (of the 15*15 = 225 resulting mathematical functions) can be omitted due to symmetry.
# There might very well be a better/more elegant way to do this than relying on some random samples rounded to a certain number of digits ;).
def determineSymmetriesProducts():

    u3, v3 = 0.02, 0.56
    # Test the symmetries using 6-fold symmetric points
    P3 = np.array([[u3, v3], [u3, 1-u3-v3], [v3, 1-u3-v3], [1-u3-v3, v3], [1-u3-v3, u3], [v3, u3]])
    print(P3)
        
    R3 = np.zeros([15, 1, 6])
    R4 = np.zeros([15, 1, 6])
    for i in range(15):
        row = np.zeros([1, 6])
        row[0][0] = nG(P3[0,0], P3[0,1])[i]
        row[0][1] = nG(P3[1,0], P3[1,1])[i]
        row[0][2] = nG(P3[2,0], P3[2,1])[i]
        row[0][3] = nG(P3[3,0], P3[3,1])[i]
        row[0][4] = nG(P3[4,0], P3[4,1])[i]
        row[0][5] = nG(P3[5,0], P3[5,1])[i]
        #print("row ", row)
        R3[i] = row

        row = np.zeros([1, 6])
        row[0][0] = ndGdu(P3[0,0], P3[0,1])[i]
        row[0][1] = ndGdu(P3[1,0], P3[1,1])[i]
        row[0][2] = ndGdu(P3[2,0], P3[2,1])[i]
        row[0][3] = ndGdu(P3[3,0], P3[3,1])[i]
        row[0][4] = ndGdu(P3[4,0], P3[4,1])[i]
        row[0][5] = ndGdu(P3[5,0], P3[5,1])[i]
        print("row ", row)
        R4[i] = row
        
    print("RRRRRR3", R3)
    print("RRRRRR4", R4)


    L = np.zeros((15, 15))

    for k in range(15):
        for l in range(15):
            #prodVec = (np.abs(R[k, 0, :] * R[l, 0, :]).round(10))
            #prodVec2 = (np.abs(R2[k, 0, :] * R2[l, 0, :]).round(10))
            prodVec3 = (np.abs(R3[k, 0, :] * R4[l, 0, :]))
            #prodVec3 = (np.abs(duR3[k, 0, :] * duR3[l, 0, :]))
            #print("prodVec3=========", prodVec3)
            L[k, l] = np.sum(prodVec3).round(10)
            #L[k, l] = (prodVec3[0]*prodVec3[1]*prodVec3[2]*prodVec3[3]*prodVec3[4]*prodVec3[5]).round(10)
            if k==1 and l==5:
                L[k, l] = 3.1432974338
            if k==1 and l==6:
                L[k, l] = 3.1432974338
                #print("p2p2===============",L[k, l])

    U, I = np.unique(L, return_index = True)
    print(U.size)
    print("U")
    print(U)
    print("I")
    print(I)
    #print(U.size)
    # ~ print(L.flatten()[I]) # Flatten handles entries row by row

    indsRowCol = np.array([ I // 15, I % 15 ])
    print("indsRowCol")
    print(indsRowCol)
    print("L[1,5]")
    print(L[1,5])
    print("L[0,0]")
    print(L[0,0])
    print("L[2,2]")
    print(L[2,2])
    print("L[1,6]")
    print(L[1,6])

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
    print("computeExactInts")
    print(indsRowCol.shape[1])
    #for k in range(indsRowCol.shape[1]):
		#F = G[indsRowCol[0, k]] * G[indsRowCol[1, k]]
        #F = (G[indsRowCol[0, k]] * dGdu[indsRowCol[1, k]]).simplify()
        
		 #~ F = dGdu[indsRowCol[0, k]] * dGdv[indsRowCol[1, k]]
        #print("k:.............", k)
        #print(F, ",", sep = '')

	# Hardcode values for exact integrals (obtained using Maple)
	# dGdu*dGdv (51)
    #quit()
    return np.array([0.0714285714285714285714286, 0., -0.00714285714285714285714286, -0.0375000000000000000000000, 0.0339285714285714285714286, 0.00982142857142857142857143, -0.00446428571428571428571429, -0.00446428571428571428571429, -0.0187500000000000000000000, -0.0107142857142857142857143, -0.0107142857142857142857143, -0.00714285714285714285714286, -0.00357142857142857142857143, -0.00357142857142857142857143, -0.00714285714285714285714286, 0., 0.0285714285714285714285714, 0.0714285714285714285714286, 0., 0., 0., 0.0375000000000000000000000, -0.0187500000000000000000000, 0., 0.0160714285714285714285714, 0.00505952380952380952380952, -0.00386904761904761904761905, -0.00863095238095238095238095, -0.0166666666666666666666667, -0.00166666666666666666666667, 0.00166666666666666666666667, 0., -0.00238095238095238095238095, -0.00363945578231292517006803, -0.00469387755102040816326531, -0.00446428571428571428571429, -0.0160714285714285714285714, 0.0428571428571428571428571, 0.0238095238095238095238095, -0.0136904761904761904761905, -0.00773809523809523809523810, -0.00863095238095238095238095, -0.00809523809523809523809524, -0.0133333333333333333333333, -0.0128571428571428571428571, -0.00738095238095238095238095, -0.00578231292517006802721088, -0.00612244897959183673469388, -0.00505952380952380952380952, 0.0404761904761904761904762, 0.0428571428571428571428571, -0.0321428571428571428571429, -0.0136904761904761904761905, -0.00386904761904761904761905, -0.00374149659863945578231293, -0.00816326530612244897959184, -0.0116666666666666666666667, -0.00976190476190476190476190, -0.00612244897959183673469388, -0.00340136054421768707482993, 0.0107142857142857142857143, -0.00714285714285714285714286, 0.00166666666666666666666667, 0.00809523809523809523809524, 0.00374149659863945578231293, -0.00340136054421768707482993, -0.00612244897959183673469388, -0.00469387755102040816326531, 0., 0.00190476190476190476190476, 0.000864073947131824870, -0.001544346055975362285, -0.00235348475879757475, -0.00172814789426364974, -0.00357142857142857142857143, -0.00166666666666666666666667, 0.0133333333333333333333333, 0.00816326530612244897959184, -0.00612244897959183673469388, -0.00578231292517006802721088, -0.00363945578231292517006803, -0.00190476190476190476190476, 0., -0.000864073947131824870, -0.003217558705929399619, -0.00308869211195072457, -0.00235348475879757475, 0.0128571428571428571428571, 0.0116666666666666666666667, -0.00976190476190476190476190, -0.00738095238095238095238095, -0.000864073947131824870, 0.000864073947131824870, 0., -0.00380952380952380952380952, -0.003217558705929399619, -0.001544346055975362285], dtype=np.longdouble)

	# ---

	# For testing whether the obtained quadratures work for all 400 products
    return np.array([], dtype=np.longdouble)

# ---

def runOptimisation():

    print("determineSymmetriesProducts")
    indsRowCol = determineSymmetriesProducts()
    print(indsRowCol)
    

    exactInts = computeExactInts(indsRowCol)
    print("exactInts")
    print(exactInts)
    print("Unique integrals:", len(np.unique(exactInts)))
    #quit()

    #single point without symmetry
    #numSinglePt = NUMOFSINGLEPOINT #
    #2 symmetric points u,v v,u
    #numTwoSymPt = NUMOFTWOSYMPOINT #
    #numThreeSymPt = NUMOFTHREESYMPOINT #
    #numFourSymPt = NUMOFFOURSYMPOINT #
    #numSixSymPt = NUMOFSIXSYMPOINT #

    #numSinglePt = NUMOFSINGLEPOINT #
    #3-fold points on vertical lines
    numOnVertLine = NUMOFTHREESYMPOINT #
    #6-fold symmetric points
    numSixPts = NUMOFSIXSYMPOINT
    numSingle = NUMOFSINGLEPOINT
    numTotal = numOnVertLine + numSixPts + numSingle
    
    #numTotal = numSinglePt + numTwoSymPt # + numThreeSymPt + numFourSymPt + numSixSymPt

    initGuess = np.empty(3*numTotal, dtype=np.longdouble)
    print("initGuess")
    print(initGuess)

    # Bounds on Pts and Wts
    Eps = 1e-3
    lowerBoundsPts = Eps + np.zeros(2*numTotal, dtype=np.longdouble) # Careful with singularities at corners of the domain!
    lowerBoundsWts = np.zeros(numTotal, dtype=np.longdouble) # -np.inf * np.ones(...)
    lowerBounds = np.concatenate((lowerBoundsPts, lowerBoundsWts))
    print("lowerBounds")
    print(lowerBounds)

    upperBoundsPts = np.ones(2*numTotal, dtype=np.longdouble) - Eps # Careful with singularities at corners of the domain!
    upperBoundsWts = np.ones(numTotal, dtype=np.longdouble) # np.inf * np.ones(...)
    upperBounds = np.concatenate((upperBoundsPts, upperBoundsWts))
    print("upperBounds")
    print(upperBounds)

    numConverged = 0
    maxConverged = 5 #5

    resTol = 2.24e-16 #2.24e-16
    varTol = 1e-18 #1e-18

    maxEvals = 5e3 #1e4 #3e4 #5e3 # To be tweaked
    markers = ['o', 'x', '*', 'D', 's']

    while (numConverged < maxConverged):

		# Set initial guess
        for k in range(numTotal):
		    # Point (u,v)... Take into account the bounds!
            initGuess[k] = Eps + (1 - 2*Eps) * np.random.rand() #
            initGuess[numTotal + k] = np.random.uniform(Eps, 1 - initGuess[k]) #
		    # Weight w
            initGuess[2*numTotal + k] = np.random.rand() / 6 # Or / 8
        print("initGuess====",initGuess)

		# ~ print(initGuess)

		# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
		# Methods are 'trf', 'dogbox' and 'lm'
        leastSq = least_squares(currentResiduals, initGuess, jac='2-point',
                                bounds=(lowerBounds, upperBounds),
                                method='trf', ftol=resTol, xtol=varTol, gtol=None,
                                max_nfev=maxEvals, args=(exactInts, indsRowCol), verbose = 1) # 2 # jac=currentJacobian

        if leastSq.success:
            currentSol = leastSq.x
            print("currentSol")
            print(currentSol)
            relErrors = currentResiduals(currentSol, exactInts, indsRowCol) #if (0 in exactInts) else currentResiduals(currentSol, exactInts, indsRowCol) / exactInts

            if np.max(np.abs(relErrors)) < 1e-16: #resTol:
                print(":: [", numConverged+1, "/", maxConverged, "] Converged after ", leastSq.nfev, " iterations", sep = '')
                print(relErrors)

                us6 = currentSol[:numSixPts]
                vs6 = currentSol[numTotal:numTotal+numSixPts]
                ws6 = currentSol[2*numTotal:(2*numTotal+numSixPts)]

                us3 = currentSol[numSixPts:numSixPts+numOnVertLine]
                vs3 = currentSol[numTotal+numSixPts:numTotal+numSixPts+numOnVertLine]
                ws3 = currentSol[2*numTotal+numSixPts:(2*numTotal+numSixPts+numOnVertLine)]

                us1 = currentSol[numSixPts+numOnVertLine:numTotal]
                vs1 = currentSol[numTotal+numSixPts+numOnVertLine:numTotal*2]
                ws1 = currentSol[2*numTotal+numSixPts+numOnVertLine:3*numTotal]

                #singleU = currentSol[numSixPts+numOnVertLine:numTotal]
                #singleV = currentSol[numTotal+numSixPts+numOnVertLine:2*numTotal]
                #singleW = currentSol[2*numTotal+numSixPts+numOnVertLine:]
			    

			    #print("-> PtsU =", list(PtsU)) # List, to add separating commas between entries
			    #print("-> PtsV =", list(PtsV))
			    #print("-> Wts =", list(Wts))
			    #print("-> Sum(Wts) =", np.sum(Wts))
                print("")
                print("POINTs:")
                print("6")
                print([us6])
                print([vs6])
                print("3")
                print([us3])
                print([1-us3-us3])
                print("1")
                print([us1])
                print([1/3])
			    #print([PtsU, PtsU, PtsV, 1-PtsU-PtsV, 1-PtsU-PtsV, PtsV])
			    #print([PtsV, 1-PtsU-PtsV, 1-PtsU-PtsV, PtsV, PtsU, PtsU])
			    #print("[Wts, Wts, Wts, Wts, Wts, Wts]")
			    #print([Wts, Wts, Wts, Wts, Wts, Wts])
                print("WEIGHTs:")
                print([ws6, ws3, ws1])
			    #uAll = np.append(np.append(us6, us3),singleU)
			    #vAll = np.append(np.append(1-us-us, us), singleV)
			    #wAll = np.append(np.append(ws, ws), singleW)
			    #print("wAll",wAll)
                uAll = np.append(np.append(us6, [us6, vs6, 1-us6-vs6, 1-us6-vs6, vs6]),
                                 np.append(us3, [1-us3-us3, us3]))
                uAll = np.append(uAll, 1/3)
                vAll = np.append(np.append(vs6, [1-us6-vs6, 1-us6-vs6, vs6, us6, us6]),
                                 np.append(us3, [us3, 1-us3-us3]))
                vAll = np.append(vAll, 1/3)
                wAll = np.append(np.append(ws6, [ws6, ws6, ws6, ws6, ws6]),
                                 np.append(ws3, [ws3, ws3]))
                wAll = np.append(wAll, ws1)
                print("uAll",uAll)
                print("vAll",vAll)
                print("wAll",wAll)

                pointsAll = []
                for point in range(NUMOFSIXSYMPOINT * 6 + NUMOFTHREESYMPOINT * 3 + numSingle):
                    pointsAll.append([uAll[point], vAll[point], 1-uAll[point]-vAll[point]])
                print("pointsAll:")
                print(pointsAll)

                pointsX = []
                pointsY = []
                for point in range(NUMOFSIXSYMPOINT * 6 + NUMOFTHREESYMPOINT * 3 + numSingle):
                    pointsX.append(np.dot(pointsAll[point], vertices[:,0]))
                    pointsY.append(np.dot(pointsAll[point], vertices[:,1]))

                #filter the points out of the triangle
                flag_negative = 0
                for x in range(NUMOFSIXSYMPOINT):
                    if(us6[x] > 0 and us6[x] < 1 and
                       vs6[x] > 0 and vs6[x] < 1 - us6[x] and
                       ws6[x] > 0):
                        print("U6 V6 W6 OK")
                    else:
                        flag_negative = 1
                for y in range(NUMOFTHREESYMPOINT):
                    if(us3[y] > 0 and us3[y] < 1 and
                       vs3[y] > 0 and vs3[y] < 1 - us3[y] and
                       ws3[y] > 0):
                        print("U3 V3 W3 OK")
                    else:
                        flag_negative = 1

                if (flag_negative == 1):
                    print("Continue")
                    continue


                NODESALL = NUMOFSIXSYMPOINT * 6 + NUMOFTHREESYMPOINT * 3 + numSingle
                NODESymeetric = NUMOFSIXSYMPOINT + NUMOFTHREESYMPOINT + numSingle
                NODESALL = NODESymeetric

########Save the results to workbook
##                book = xlwt.Workbook() # 新建工作簿
##                table = book.add_sheet('play',cell_overwrite_ok=True) # 如果对同一单元格重复操作会发生overwrite Exception，cell_overwrite_ok为可覆盖
##                style = xlwt.XFStyle() # 新建样式
##                font = xlwt.Font() #新建字体
##                font.name = 'Times New Roman'
##                font.bold = True
##                style.font = font # 将style的字体设置为font
##
##                
##                table.write(1,0,'Test',style)
##                for y in range(NUMOFSIXSYMPOINT):
##                    table.write(1, y, str(us6[y]))
##                for y in range(NUMOFTHREESYMPOINT):
##                    table.write(1, y+NUMOFSIXSYMPOINT, str(us3[y]))
##                for y in range(numSingle):
##                    table.write(1, y+NUMOFSIXSYMPOINT+NUMOFTHREESYMPOINT, str(1/3))
##
##                for y in range(NUMOFSIXSYMPOINT):
##                    table.write(1, y+NODESALL, str(vs6[y]))
##                for y in range(NUMOFTHREESYMPOINT):
##                    table.write(1, y+NODESALL+NUMOFSIXSYMPOINT, str(1-us3[y]-us3[y]))
##                for y in range(numSingle):
##                    table.write(1, y+NODESALL+NUMOFSIXSYMPOINT+NUMOFTHREESYMPOINT, str(1/3))
##    
##
##                for y in range(NUMOFSIXSYMPOINT):
##                    table.write(1, y+NODESALL+NODESALL, str(ws6[y]))
##                for y in range(NUMOFTHREESYMPOINT):
##                    table.write(1, y+NODESALL+NODESALL+NUMOFSIXSYMPOINT, str(ws3[y]))
##                for y in range(numSingle):
##                    table.write(1, y+NODESALL+NODESALL+NUMOFSIXSYMPOINT+NUMOFTHREESYMPOINT, str(ws1[y]))
##
##                book.save(filename_or_stream='GGuSymmetric631.xls') # 一定要保存

                

                data = xlrd.open_workbook('GGuSymmetric631.xls',formatting_info=True)
                excel = copy(wb=data) # 完成xlrd对象向xlwt对象转换
                excel_table = excel.get_sheet(0) # 获得要操作的页
                table = data.sheets()[0]
                nrows = table.nrows # 获得行数
                ncols = table.ncols # 获得列数
                values = []
                for y in range(NUMOFSIXSYMPOINT):
                    values.append(str(us6[y]))
                for y in range(NUMOFTHREESYMPOINT):
                    values.append(str(us3[y]))
                for y in range(numSingle):
                    values.append(str(1/3))

                for y in range(NUMOFSIXSYMPOINT):
                    values.append(str(vs6[y]))
                for y in range(NUMOFTHREESYMPOINT):
                    values.append(str(1-us3[y]-us3[y]))
                for y in range(numSingle):
                    values.append(str(1/3))
    

                for y in range(NUMOFSIXSYMPOINT):
                    values.append(str(ws6[y]))
                for y in range(NUMOFTHREESYMPOINT):
                    values.append(str(ws3[y]))
                for y in range(numSingle):
                    values.append(str(ws1[y]))

                col = 0
                for value in values:
                    excel_table.write(nrows,col,value) # 因为单元格从0开始算，所以row不需要加一
                    col = col+1
                excel.save('GGuSymmetric631.xls')


				# Plot stuff
##			    plt.scatter([PtsU, 1-PtsU, 1-PtsV, 1-PtsV, 1-PtsU, PtsU, PtsV, PtsV], [PtsV, PtsV, PtsU, 1-PtsU, 1-PtsV, 1-PtsV, 1-PtsU, PtsU], s=1e4*np.abs([Wts, Wts, Wts, Wts, Wts, Wts, Wts, Wts]), color=plt.cm.hsv(numConverged/maxConverged), edgecolors='none', alpha=0.5, label=str(numConverged)) # 0.2 # 0.01
		#plt.scatter([PtsU, PtsU, PtsV, 1-PtsU-PtsV, 1-PtsU-PtsV, PtsV], [PtsV, 1-PtsU-PtsV, 1-PtsU-PtsV, PtsV, PtsU, PtsU], s=1e4*np.abs([Wts, Wts, Wts, Wts, Wts, Wts]), markers = ['o', 'x', '*', 'D', 's'], color=plt.cm.hsv(numConverged/maxConverged), edgecolors='none', alpha=0.5, label=str(numConverged))
				#color='tab:blue'
                plt.scatter(pointsX, pointsY,
                            s=1e4*np.abs([wAll/2]),
                            marker=markers[0],
                            color=plt.cm.hsv(numConverged/maxConverged),
                            #color='black',
                            edgecolors='none', alpha=1,
                            label=str(numConverged))

                numConverged += 1

            else:
				# ~ pass
                print(".. Successful, but minimal (absolute) error not low enough (", np.max(np.abs(relErrors)), ")", sep = '')

        else:
			# ~ pass
            print("Not successful...")

    # Plot commands
    plt.axis('square')
    plt.gca().set(xlim=(-1, 1))
    plt.gca().set(ylim=(-0.1, pow(3, 0.5)))
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) # ["0", "1", ...]
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
