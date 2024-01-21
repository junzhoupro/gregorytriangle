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

print("nG")
print(nG, ndGdu)

# ~ print( G.subs([(u, 0.5), (v, 0.25)]) )
evalG = ndGdu(0.5, 0.25)

# ~ print(evalG) # This should be a bit faster than G.subs(...)
print(evalG.shape)

NUMOFSINGLEPOINT = 0 #
# Set the number of 3-fold symmetric points
NUMOFTHREEPOINT = 2 #
# Set the number of 6-fold symmetric points
NUMOFSIXPOINT = 3 #

# 3 points v0, v1, v2， counterclockwise
# v0
vertices = np.array([[-1, 0], [1, 0], [0, pow(3, 0.5)]])
print("vertices")
print(vertices.sum(axis=0))
print(vertices.sum(axis=0)[0])
print(vertices.sum(axis=0)[1])

# Use 6-fold symmetry here:
# Because I saw the symmetry in the plots of maple
def evalFuncSix(uVals, vVals, indsRowCol):
    u, v = uVals, vVals
    F = nG
    #print("F")
    #print(F)
    #print(F(u, v))
    evalsF = np.array([ F(u, v), F(u, 1-u-v), F(v, 1-u-v), F(1-u-v, v), F(1-u-v, u), F(v, u) ])[:,:,0,:]
    #print("evalsF")
    #print(evalsF)
    #print("evalsF[:, indsRowCol[0, :], :]")
    #print(evalsF[:, indsRowCol[0, :], :])
    #print("evalsF[:, indsRowCol[1, :], :]")
    #print(evalsF[:, indsRowCol[1, :], :])
    funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsF[:, indsRowCol[1, :], :] # Now (4 or 8, 33, numPtsWts)
    #print("funcProducts")
    #print(funcProducts)
    #quit()
	# ~ funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsH[:, indsRowCol[1, :], :] # Now (4 or 8, 33, numPtsWts)
	# ~ print(funcProducts.shape). For testing with G, comment out the multiplication

	#print(funcProducts.shape)
    return np.sum(funcProducts, axis=0)


# Use 3-fold symmetry here:
def evalFuncsThree(uVal, vVal, indsRowCol):
    u, v = uVal, vVal
    F = nG
    #print("F")
    #print(F)
    #print("u", u)
    #evalF = np.array([ F(u, u) ])[:,:,0,:]
    numOfVals = len(u)
    evalsF = np.array([ F(u, u), F(1-u-u, u), F(u, 1-u-u) ])[:,:,0,:]
    #evalsF = np.zeros([1, 15, numOfVals])
    #for j in range(15):
     #   row = np.zeros(numOfVals)
      #  for k in range(numOfVals):
       #     row[k] = F(u[k], u[k])[j]
        #evalsF[0][j] = row
    #funcProducts = evalsF[:, indsRowCol, :]
    funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsF[:, indsRowCol[1, :], :]
    #print("evalsF")
    #print(evalsF)
    #print("funcProducts+++++++++")
    #print(funcProducts)
    #print("np.sum(funcProducts, axis=0)+++++++++")
    #print(np.sum(funcProducts, axis=0))
    return np.sum(funcProducts, axis=0)

# This function is to evaluate the median point
def evalFuncsSingle(uVals, vVals, indsRowCol):
    u, v = uVals, vVals
    F = nG
    numOfVals = len(uVals)

    evalsF = np.array([ F(1/3, 1/3) ])[:,:,0,:]
            
	#evalsF = np.array([ F(u, v), F(u, 1-u-v), F(v, 1-u-v), F(1-u-v, v), F(1-u-v, u), F(v, u) ])[:,:,0,:]
	
	#print("evalsF===", evalsF)
	#evalsH = np.array([ H(u, v), H(1-u, v), H(1-v, u), H(1-v, 1-u), H(1-u, 1-v), H(u, 1-v), H(v, 1-u), H(v, u) ])[:,:,0,:]

	#funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsF[:, indsRowCol[1, :], :] # Now (4 or 8, 33, numPtsWts)
    funcProducts = evalsF[:, indsRowCol[0, :], :] * evalsF[:, indsRowCol[1, :], :] # Now (4 or 8, 33, numPtsWts)
	# ~ print(funcProducts.shape). For testing with G, comment out the multiplication

	#print(funcProducts.shape)
    return np.sum(funcProducts, axis=0)


# This functions is to calculate residuals between the guess and the true integrals
# optVars are the Pts (for now all using 3-fold symmetry and centre one) and Wts
# exactInts contains the array of exact integrals
# numTwoSymPt: how many pairs do we have a 2-fold symmetric points
# us6, vs6, ws6 are variables to store (u,v,weights) of the possible 6-fold symmetric points
# funcEvals6: evaluation of the possible 2-fold symmetric points
# us3, vs3, ws3 are variables to store (u,v,weights) of the possible 3-fold symmetric points
# funcEvals3: evaluation of the possible 2-fold symmetric points
# singleU, singleV, singleW are variables to store (u,v,weights) of the possible single points
# funcEvalSingle: evaluation of the possible single points
def currentResiduals(optVars, exactInts, indsRowCol):
    numPtsWts = len(optVars) // 3
    numThreePoints = NUMOFTHREEPOINT #
    numSixPoints = NUMOFSIXPOINT #

    numOnVertLine = NUMOFTHREEPOINT # u, v = v, u
    numSixPts = NUMOFSIXPOINT
    numSingle = NUMOFSINGLEPOINT
    numTotal = numOnVertLine + numSixPts

    us6 = optVars[:numSixPts]
    vs6 = optVars[numTotal:numTotal+numSixPts]
    ws6 = optVars[2*numTotal:(2*numTotal+numSixPts)]

    us3 = optVars[numSixPts:numSixPts+numOnVertLine]
    vs3 = optVars[numTotal+numSixPts:numTotal+numSixPts+numOnVertLine]
    ws3 = optVars[2*numTotal+numSixPts:(2*numTotal+numSixPts+numOnVertLine)]

	#print("optVars>>>>>>>>>>>>", optVars)
	#print("us6>>>>>>>>>>>>>>>>", us6)
	#print("vs6>>>>>>>>>>>>>>>>", vs6)
	#print("ws6>>>>>>>>>>>>>>>>", ws6)
	#print("us3>>>>>>>>>>>>>>>>", us3)
	#print("vs3>>>>>>>>>>>>>>>>", vs3)
	#print("ws3>>>>>>>>>>>>>>>>", ws3)
	#print("singleU>>>>>>>>>>>>", singleU)
	#print("singleV>>>>>>>>>>>>", singleV)
	#print("singleW>>>>>>>>>>>>", singleW)

	# Pts are the first and second "third" of optVars
	#funcEvals = evalFuncs(optVars[:numPtsWts], optVars[numPtsWts:2*numPtsWts], indsRowCol)
	# ~ print(funcEvals.shape)
	# ~ print(funcEvals)
    funcEvals6 = evalFuncSix(us6,
                           vs6,
                           indsRowCol)
    funcEvals3 = evalFuncsThree(us3,
                                vs3,
                                indsRowCol)


	# Weights are the last "third" of optVars
	#currentWeights = optVars[2*numPtsWts:]
    return funcEvals6.dot(ws6) + funcEvals3.dot(ws3) - exactInts

	#return funcEvals.dot(currentWeights) - exactInts


# ---
# The Python function determineSymmetries is a somewhat ad-hoc function I wrote to (numerically)
# figure out which products (of the 15*15 = 225 resulting mathematical functions) can be omitted due to symmetry.
# There might very well be a better/more elegant way to do this than relying on some random samples rounded to a certain number of digits ;).
def determineSymmetriesProducts():

    u3, v3 = 0.23, 0.58
    # Test the symmetries using 6-fold symmetric points
    P = np.array([[u3, v3], [u3, 1-u3-v3], [v3, 1-u3-v3], [1-u3-v3, v3], [1-u3-v3, u3], [v3, u3]])
    print(P)
        
    R = nG(P[:,0], P[:,1])

    L = np.zeros((15, 15))

    for k in range(15):
        for l in range(15):
            prodVec = (np.abs(R[k, 0, :] * R[l, 0, :]).round(10))
            #prodVec2 = (np.abs(R2[k, 0, :] * R2[l, 0, :]).round(10))
            #prodVec3 = (np.abs(R3[k, 0, :] * R4[l, 0, :]))
            #prodVec3 = (np.abs(duR3[k, 0, :] * duR3[l, 0, :]))
            #print("prodVec3=========", prodVec3)
            L[k, l] = np.sum(prodVec).round(10)
            #L[k, l] = (prodVec3[0]*prodVec3[1]*prodVec3[2]*prodVec3[3]*prodVec3[4]*prodVec3[5]).round(10)


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
	#	F = G[indsRowCol[0, k]] * G[indsRowCol[1, k]]
		#F = (dGdu[indsRowCol[0, k]] * dGdv[indsRowCol[1, k]]).simplify()
		# ~ F = dGdu[indsRowCol[0, k]] * dGdv[indsRowCol[1, k]]
		#print("k:.............", k)
		#print(F, ",", sep = '')

	# Hardcode values for exact integrals (obtained using Maple)
	# G*G (24)
    return np.array([0.0178571428571428571428571, 0.000892857142857142857142857, 0.00793650793650793650793651, 0.00317460317460317460317460, 0.000496031746031746031746032, 0.00198412698412698412698413, 0.00113378684807256235827664, 0.000453514739229024943310658, 0.00857142857142857142857143, 0.00321428571428571428571429, 0.00136904761904761904761905, 0.000773809523809523809523810, 0.00642857142857142857142857, 0.00238095238095238095238095, 0.00142857142857142857142857, 0.000952380952380952380952381, 0.000714285714285714285714286, 0.000884353741496598639455782, 0.00197278911564625850340136, 0.00108843537414965986394558, 0.000816326530612244897959184, 0.000650836440427161578, 0.000573653355491205768, 0.000981816620797328217])
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

    #3-fold points on vertical lines
    numOnVertLine = NUMOFTHREEPOINT #
    #6-fold symmetric points
    numSixPts = NUMOFSIXPOINT
    numTotal = numOnVertLine + numSixPts

    initGuess = np.empty(3*numTotal, dtype=np.longdouble)
    print("initGuess")
    print(initGuess)

    # Bounds on Pts and Wts
    Eps = 1e-3
    lowerBoundsPts = np.zeros(2*numTotal, dtype=np.longdouble) # Careful with singularities at corners of the domain!
    lowerBoundsWts = np.ones(numTotal, dtype=np.longdouble) # -np.inf * np.ones(...)
    lowerBounds = np.concatenate((lowerBoundsPts, lowerBoundsWts.dot(-1)))
    print("lowerBounds")
    print(lowerBounds)

    upperBoundsPts = np.ones(2*numTotal, dtype=np.longdouble) # Careful with singularities at corners of the domain!
    upperBoundsWts = np.ones(numTotal, dtype=np.longdouble) # np.inf * np.ones(...)
    upperBounds = np.concatenate((upperBoundsPts, upperBoundsWts))
    print("upperBounds")
    print(upperBounds)

    numConverged = 0
    maxConverged = 10 #5

    resTol = 2.24e-16 #2.24e-16
    varTol = 1e-18 #1e-18

    maxEvals = 5e3 #1e4 #3e4 #5e3 # To be tweaked
    markers = ['o', 'x', '*', 'D', 's']

    while (numConverged < maxConverged):
        # Set initial guess
        for k in range(numTotal):
            # Point (u,v)... Take into account the bounds!
            #u
            initGuess[k] = Eps + (1 - 2*Eps) * np.random.rand() #
            #v: generate random number between [0, u), make sure w > 0, inside the triangle
            initGuess[numTotal + k] = np.random.uniform(0, initGuess[k]) #
            # Weight w
            initGuess[2*numTotal + k] = np.random.rand() / 6 # Or / 8

        print("initGuess=====", initGuess)

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

            if np.max(np.abs(relErrors)) < 1e-16: #resTol:
                print(":: [", numConverged+1, "/", maxConverged, "] Converged after ", leastSq.nfev, " iterations", sep = '')
                print(relErrors)

			    #PtsU = currentSol[:numPtsWts]
			    #PtsV = currentSol[numPtsWts:2*numPtsWts]
			    #Wts = currentSol[2*numPtsWts:]

			    # Sort
			    #PtsU = PtsU[Wts.argsort()]
			    #PtsV = PtsV[Wts.argsort()]
			    #Wts.sort()

                us6 = currentSol[:numSixPts]
                vs6 = currentSol[numTotal:numTotal+numSixPts]
                ws6 = currentSol[2*numTotal:(2*numTotal+numSixPts)]

                us3 = currentSol[numSixPts:numSixPts+numOnVertLine]
                vs3 = currentSol[numTotal+numSixPts:numTotal+numSixPts+numOnVertLine]
                ws3 = currentSol[2*numTotal+numSixPts:(2*numTotal+numSixPts+numOnVertLine)]

			    

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
			    #print([PtsU, PtsU, PtsV, 1-PtsU-PtsV, 1-PtsU-PtsV, PtsV])
			    #print([PtsV, 1-PtsU-PtsV, 1-PtsU-PtsV, PtsV, PtsU, PtsU])
			    #print("[Wts, Wts, Wts, Wts, Wts, Wts]")
			    #print([Wts, Wts, Wts, Wts, Wts, Wts])
                print("WEIGHTs:")
                print([ws6, ws3])
			    #uAll = np.append(np.append(us6, us3),singleU)
			    #vAll = np.append(np.append(1-us-us, us), singleV)
			    #wAll = np.append(np.append(ws, ws), singleW)
			    #print("wAll",wAll)
                uAll = np.append(np.append(us6, [us6, vs6, 1-us6-vs6, 1-us6-vs6, vs6]),
                                 np.append(us3, [1-us3-us3, us3]))
                vAll = np.append(np.append(vs6, [1-us6-vs6, 1-us6-vs6, vs6, us6, us6]),
                                 np.append(us3, [us3, 1-us3-us3]))
                wAll = np.append(np.append(ws6, [ws6, ws6, ws6, ws6, ws6]),
                                 np.append(ws3, [ws3, ws3]))
                print("uAll",uAll)
                print("vAll",vAll)
                print("wAll",wAll)

                pointsAll = []
                for point in range(NUMOFSIXPOINT * 6 + NUMOFTHREEPOINT * 3):
                    pointsAll.append([uAll[point], vAll[point], 1-uAll[point]-vAll[point]])
                print("pointsAll:")
                print(pointsAll)

                pointsX = []
                pointsY = []
                for point in range(NUMOFSIXPOINT * 6 + NUMOFTHREEPOINT * 3):
                    pointsX.append(np.dot(pointsAll[point], vertices[:,0]))
                    pointsY.append(np.dot(pointsAll[point], vertices[:,1]))

                #filter the points out of the triangle
                flag_negative = 0
                for x in range(NUMOFSIXPOINT):
                    if(us6[x] > 0 and us6[x] < 1 and
                       vs6[x] > 0 and vs6[x] < 1 - us6[x] and
                       ws6[x] > 0):
                        print("U6 V6 W6 OK")
                    else:
                        flag_negative = 1
                for y in range(NUMOFTHREEPOINT):
                    if(us3[y] > 0 and us3[y] < 1 and
                       vs3[y] > 0 and vs3[y] < 1 - us3[y] and
                       ws3[y] > 0):
                        print("U3 V3 W3 OK")
                    else:
                        flag_negative = 1

                if (flag_negative == 1):
                    print("Continue")
                    continue

                NODESALL = NUMOFSIXPOINT * 6 + NUMOFTHREEPOINT * 3
                NODESymeetric = NUMOFSIXPOINT + NUMOFTHREEPOINT
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
##                for y in range(NUMOFSIXPOINT):
##                    table.write(1, y, str(us6[y]))
##                for y in range(NUMOFTHREEPOINT):
##                    table.write(1, y+NUMOFSIXPOINT, str(us3[y]))
##
##                for y in range(NUMOFSIXPOINT):
##                    table.write(1, y+NODESALL, str(vs6[y]))
##                for y in range(NUMOFTHREEPOINT):
##                    table.write(1, y+NODESALL+NUMOFSIXPOINT, str(1-us3[y]-us3[y]))
##    
##
##                for y in range(NUMOFSIXPOINT):
##                    table.write(1, y+NODESALL+NODESALL, str(ws6[y]))
##                for y in range(NUMOFTHREEPOINT):
##                    table.write(1, y+NODESALL+NODESALL+NUMOFSIXPOINT, str(ws3[y]))
##
##                book.save(filename_or_stream='GGSymmetric.xls') # 一定要保存

                

##                data = xlrd.open_workbook('GGSymmetric.xls',formatting_info=True)
##                excel = copy(wb=data) # 完成xlrd对象向xlwt对象转换
##                excel_table = excel.get_sheet(0) # 获得要操作的页
##                table = data.sheets()[0]
##                nrows = table.nrows # 获得行数
##                ncols = table.ncols # 获得列数
##                values = []
##                for y in range(NUMOFSIXPOINT):
##                    values.append(str(us6[y]))
##                for y in range(NUMOFTHREEPOINT):
##                    values.append(str(us3[y]))
##
##                for y in range(NUMOFSIXPOINT):
##                    values.append(str(vs6[y]))
##                for y in range(NUMOFTHREEPOINT):
##                    values.append(str(1-us3[y]-us3[y]))
##    
##
##                for y in range(NUMOFSIXPOINT):
##                    values.append(str(ws6[y]))
##                for y in range(NUMOFTHREEPOINT):
##                    values.append(str(ws3[y]))
##
##                col = 0
##                for value in values:
##                    excel_table.write(nrows,col,value) # 因为单元格从0开始算，所以row不需要加一
##                    col = col+1
##                excel.save('GGSymmetric.xls')
##               
                plt.scatter(pointsX, pointsY,
                            s=1e4*np.abs([wAll/4]),
                            marker=markers[0],
                            color=plt.cm.hsv(numConverged/maxConverged),
                            edgecolors='none', alpha=1,
                            label=str(numConverged))

				# Plot stuff
##			    plt.scatter([PtsU, 1-PtsU, 1-PtsV, 1-PtsV, 1-PtsU, PtsU, PtsV, PtsV], [PtsV, PtsV, PtsU, 1-PtsU, 1-PtsV, 1-PtsV, 1-PtsU, PtsU], s=1e4*np.abs([Wts, Wts, Wts, Wts, Wts, Wts, Wts, Wts]), color=plt.cm.hsv(numConverged/maxConverged), edgecolors='none', alpha=0.5, label=str(numConverged)) # 0.2 # 0.01
                #plt.scatter([us6, us6, vs6, 1-us6-vs6, 1-us6-vs6, vs6],
                 #           [vs6, 1-us6-vs6, 1-us6-vs6, vs6, us6, us6],
                  #          s=1e4*np.abs([ws6, ws6, ws6, ws6, ws6, ws6]),
                   #         marker=markers[0],
                    #        color=plt.cm.hsv(numConverged/maxConverged),
                     #       edgecolors='none', alpha=0.5, label=str(numConverged))
			    
                #plt.scatter([us3, 1-us3-us3, us3],
                 #           [us3, us3, 1-us3-us3],
                  #          s=1e4*np.abs([ws3, ws3, ws3]),
                   #         marker=markers[0],
                    #        color=plt.cm.hsv(numConverged/maxConverged),
                     #       edgecolors='none', alpha=0.5, label=str(numConverged))
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
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) # ["0", "1", ...]
    plt.axis('off')
    
    plt.plot([-1, 0], [0, pow(3, 0.5)], color='black', linewidth=0.7)
    plt.plot([1, 0], [0, pow(3, 0.5)], color='black', linewidth=0.7)
    plt.plot([1, -1], [0, 0], color='black', linewidth=0.7)

    plt.gcf().canvas.manager.set_window_title("Gregory product quadrature experiments, numPtsWts = " + str(numTotal))
    plt.show()

    


# ---

if __name__ == "__main__":
    runOptimisation()
