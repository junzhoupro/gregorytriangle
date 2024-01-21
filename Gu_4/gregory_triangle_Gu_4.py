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
NUMOFTWOSYMPOINT = 0 #

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

# Use general node here
def evalFuncsSingle(uVal, vVal, indsRowCol):

    u, v = uVal, vVal
    F = ndGdu

    numOfVals = len(u)

    evalsF = np.zeros([1, 15, numOfVals])
    for j in range(15):
        row = np.zeros(numOfVals)
        for k in range(numOfVals):
            #row[k] = F(u[k], (1-2*u[k]))[j]
            row[k] = F(u[k], v[k])[j]
        evalsF[0][j] = row
    funcProducts = evalsF[:, indsRowCol, :]

    return np.sum(funcProducts, axis=0)

# ---

# optVars are the Pts and Wts
# exactInts contains the array of exact integrals
def currentResiduals(optVars, exactInts, indsRowCol):
    #print("indsRowCol")
    #print(indsRowCol)
    #  u u ... v v ... w w....
    numSinglePt = NUMOFSINGLEPOINT #
    numTwoSymPt = NUMOFTWOSYMPOINT #
    numTotal = numSinglePt + numTwoSymPt

	
    singleU = optVars[:numTotal]
    singleV = optVars[numTotal:2*numTotal]
    singleW = optVars[2*numTotal:]

    funcEvalSingle = evalFuncsSingle(singleU, singleV, indsRowCol)



    return funcEvalSingle.dot(singleW) - exactInts

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
    indexSort = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

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

    return np.array([0.250000000000000000000000, 0., -0.250000000000000000000000, 0., 0.250000000000000000000000, 0.250000000000000000000000, -0.250000000000000000000000, -0.250000000000000000000000, 0., 0., 0., 0., 0., 0., 0.])
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
    numTotal = numSinglePt #+ numTwoSymPt
    #initGuess = np.empty(3*numPtsWts, dtype=np.longdouble)
    initGuess = np.empty(3*numSinglePt, dtype=np.longdouble)
    print("initGuess")
    print(initGuess)
    #  u u... v v ... w w....

    # Bounds on Pts and Wts
    Eps = 1e-3
    lowerBoundsPts = Eps + np.zeros(2*(numSinglePt), dtype=np.longdouble) # Careful with singularities at corners of the domain!
    lowerBoundsWts = np.ones((numSinglePt), dtype=np.longdouble) # -np.inf * np.ones(...)
    lowerBounds = np.concatenate((lowerBoundsPts, lowerBoundsWts.dot(0)))
    print("lowerBounds")
    print(lowerBounds)

    upperBoundsPts = np.ones(2*(numSinglePt), dtype=np.longdouble) - Eps # Careful with singularities at corners of the domain!
    upperBoundsWts = np.ones((numSinglePt), dtype=np.longdouble) # np.inf * np.ones(...)
    upperBounds = np.concatenate((upperBoundsPts, upperBoundsWts))
    print("upperBounds")
    print(upperBounds)

    numConverged = 0
    maxConverged = 50 #5

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
                
                singleU = currentSol[:numTotal]
                singleV = currentSol[numTotal:2*numTotal]
                singleW = currentSol[2*numTotal:]


                print("")
                print("POINTs:")

                print([singleU])
			    #print([1-us-us, us, 1-singleU-singleU])
                print([singleV])
                print("WEIGHTs:")
                print([singleW])
                uAll = singleU
		#vAll = np.append(np.append(1-us-us, us), 1-singleU-singleU)
                vAll = singleV
                wAll = singleW
                print("uAll",uAll)
                print("vAll",vAll)
                print("wAll",wAll)



                flag_negative = 0

                for y in range(NUMOFSINGLEPOINT):
                    if(singleU[y] > 0 and singleU[y] < 1 and singleV[y] > 0 and singleV[y] < 1):
                        print("U1 V1 OK")
                    else:
                        flag_negative = 1
                        
                for y in range(NUMOFSINGLEPOINT):
                    if(1-singleU[y]-singleV[y] > 0 and 1-singleU[y]-singleV[y] < 1):
                        print("WW1 OK")
                    else:
                        flag_negative = 1

                for y in range(NUMOFSINGLEPOINT):
                    if(singleW[y] > 0 and singleW[y] < 1):
                        print("W1 OK")
                    else:
                        flag_negative = 1

                if (flag_negative == 1):
                    print("Continue")
                    continue
                #target: 0.1 0.1 0.6 0.6
                #0.1 0.6 0.1 0.6 =
                if (vAll[1] - vAll[0] > 0.1 and vAll[3] - vAll[2] > 0.1):
                    vAll[1], vAll[2] = vAll[2], vAll[1]
                    uAll[1], uAll[2] = uAll[2], uAll[1]
                    wAll[1], wAll[2] = wAll[2], wAll[1]
                    
                #0.1 0.6 0.6 0.1 =
                if (vAll[1] - vAll[0] > 0.1 and vAll[2] - vAll[3] > 0.1):
                    vAll[1], vAll[3] = vAll[3], vAll[1]
                    uAll[1], uAll[3] = uAll[3], uAll[1]
                    wAll[1], wAll[3] = wAll[3], wAll[1]
                #0.6 0.6 0.1 0.1
                if (vAll[0] - vAll[2] > 0.1 and vAll[1] - vAll[3] > 0.1):
                    vAll[0], vAll[2] = vAll[2], vAll[0]
                    uAll[0], uAll[2] = uAll[2], uAll[0]
                    wAll[0], wAll[2] = wAll[2], wAll[0]
                    vAll[1], vAll[3] = vAll[3], vAll[1]
                    uAll[1], uAll[3] = uAll[3], uAll[1]
                    wAll[1], wAll[3] = wAll[3], wAll[1]
                #0.6 0.1 0.6 0.1 =
                if (vAll[0] - vAll[1] > 0.1 and vAll[2] - vAll[3] > 0.1):
                    vAll[0], vAll[3] = vAll[3], vAll[0]
                    uAll[0], uAll[3] = uAll[3], uAll[0]
                    wAll[0], wAll[3] = wAll[3], wAll[0]
                #0.6 0.1 0.1 0.6 =
                if (vAll[0] - vAll[1] > 0.1 and vAll[3] - vAll[2] > 0.1):
                    vAll[0], vAll[2] = vAll[2], vAll[0]
                    uAll[0], uAll[2] = uAll[2], uAll[0]
                    wAll[0], wAll[2] = wAll[2], wAll[0]

                #0.1 0.6
                #if (uAll[1] > uAll[0]):
                 #   uAll[0], uAll[1] = uAll[1], uAll[0]
                  #  vAll[0], vAll[1] = vAll[1], vAll[0]
                   # wAll[0], wAll[1] = wAll[1], wAll[0]

                color_flag = 'blue'
                if (uAll[0] > 0.78 or uAll[1] > 0.78):
                    color_flag = 'red'
                #print("max([uAll[0], uAll[1]])")
                #print(max([uAll[0], uAll[1]]))

##                book = xlwt.Workbook() # create worksheet
##                table = book.add_sheet('play',cell_overwrite_ok=True) # 如果对同一单元格重复操作会发生overwrite Exception，cell_overwrite_ok为可覆盖
##                style = xlwt.XFStyle() # 新建样式
##                font = xlwt.Font() #新建字体
##                font.name = 'Times New Roman'
##                font.bold = True
##                style.font = font # 将style的字体设置为font
##
##                table.write(1,0,'Test',style)
##                for y in range(NUMOFSINGLEPOINT):
##                    table.write(1, y, str(uAll[y]))
##
##                for y in range(NUMOFSINGLEPOINT):
##                    table.write(1, y+NUMOFSINGLEPOINT, str(vAll[y]))
##    
##
##                for y in range(NUMOFSINGLEPOINT):
##                    table.write(1, y+NUMOFSINGLEPOINT+NUMOFSINGLEPOINT, str(wAll[y]))
##
##                book.save(filename_or_stream='Gu4_100.xls') # save

                data = xlrd.open_workbook('Gu4_100.xls',formatting_info=True)
                excel = copy(wb=data) # 完成xlrd对象向xlwt对象转换
                excel_table = excel.get_sheet(0) # 获得要操作的页
                table = data.sheets()[0]
                nrows = table.nrows # 获得行数
                ncols = table.ncols # 获得列数
                values = []
                for y in range(NUMOFSINGLEPOINT):
                    values.append(str(uAll[y]))

                for y in range(NUMOFSINGLEPOINT):
                    values.append(str(vAll[y]))
    

                for y in range(NUMOFSINGLEPOINT):
                    values.append(str(wAll[y]))

                col = 0
                for value in values:
                    excel_table.write(nrows,col,value) # 因为单元格从0开始算，所以row不需要加一
                    col = col+1
                excel.save('Gu4_100.xls')
                
                pointsAll = []
                for point in range(NUMOFSINGLEPOINT):
                    pointsAll.append([uAll[point], vAll[point], 1-uAll[point]-vAll[point]])


                #symmetric rule
                uAll2 = 1-singleU-singleV
                vAll2 = singleV
                wAll2 = singleW
                print("uAll",uAll)
                print("vAll",vAll)
                print("wAll",wAll)

                pointsAll2 = []
                for point in range(NUMOFSINGLEPOINT):
                    pointsAll2.append([uAll2[point], vAll2[point], 1-uAll2[point]-vAll2[point]])
                

                
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
                            #color=plt.cm.hsv(numConverged/maxConverged),
                            color=plt.cm.hsv(max([uAll[0], uAll[1]])),
                            #color='black',
                            edgecolors='none', alpha=0.5,
                            label=str(numConverged))


                numConverged += 1

            else:
                print(".. Successful, but minimal (absolute) error not low enough (", np.max(np.abs(relErrors)), ")", sep = '')

        else:
            print("Not successful...")

    # Plot commands
    plt.axis('square')
    plt.gca().set(xlim=(-1, 1))
    plt.gca().set(ylim=(-0.1, pow(3, 0.5)))

    plt.axis('off')
    
    plt.plot([-1, 0], [0, pow(3, 0.5)], color='black', linewidth=0.7)
    plt.plot([1, 0], [0, pow(3, 0.5)], color='black', linewidth=0.7)
    plt.plot([1, -1], [0, 0], color='black', linewidth=0.7)

    plt.subplots_adjust(right=0.98, left=0.02, top=0.98, bottom=0.01)
    plt.show()


    


# ---

if __name__ == "__main__":
    runOptimisation()
