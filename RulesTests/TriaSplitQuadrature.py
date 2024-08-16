# Implemented by P.J. Barendrecht, 17 June 2024

import numpy as np

import sympy as sp
u, v = sp.symbols('u v')

import matplotlib.pyplot as plt
tableauCols = plt.get_cmap("tab10")

# ---

def F1(x, y):
    return x * (x + 7)**2 * (x - 1)**4 * y**3 * (y + 4) * (y + 1)**3

exactIntF1 = sp.integrate(u * (u + 7)**2 * (u - 1)**4 * v**3 * (v + 4) * (v + 1)**3, (v, 0, 1 - u)).simplify()
exactIntF1 = float( sp.N( sp.integrate(exactIntF1, (u, 0, 1)) ) )
print(exactIntF1)

def F2(x, y):
    return np.sin(x) * np.cos(y)

exactIntF2 = sp.integrate(sp.sin(u) * sp.cos(v), (v, 0, 1 - u)).simplify()
exactIntF2 = float( sp.N( sp.integrate(exactIntF2, (u, 0, 1)) ) )
print(exactIntF2)

def F3(x, y):
    return np.log(x**2 + y**2 + 1)

exactIntF3 = sp.integrate(sp.log(u**2 + v**2 + 1), (v, 0, 1 - u)).simplify()
exactIntF3 = float( 0.1377089500156866296765085316883088856684 ) # sp.N( sp.integrate(exactIntF3, (u, 0, 1)) ) # SymPy has issues with this integral... Thanks Jiri for the value
print(exactIntF3)

# ---

# Gauss-Legendre on [0, 1]
quadPts = [(3 - np.sqrt(3)) / 6, (3 + np.sqrt(3)) / 6]
quadWts = [1/2, 1/2]

# TP
quadQuadPts = [np.array([i, j]) for i in quadPts for j in quadPts]
quadQuadWts = [i * j for i in quadWts for j in quadWts]

# ---

class Quad:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.Id = 0

    def parametricToCartesian(self, xi, eta):
        # Bilinear map
        return (1 - xi) * (1 - eta) * self.A + xi * (1 - eta) * self.B + xi * eta * self.C + (1 - xi) * eta * self.D

    def getArea(self):
        # Using the shoelace formula
        return 0.5 * np.abs(
            (self.A[0] * self.B[1] - self.B[0] * self.A[1]) +
            (self.B[0] * self.C[1] - self.C[0] * self.B[1]) +
            (self.C[0] * self.D[1] - self.D[0] * self.C[1]) +
            (self.D[0] * self.A[1] - self.A[0] * self.D[1])
        )

    def splitCatMark(self):
        E = 0.5 * (self.A + self.B)
        F = 0.5 * (self.B + self.C)
        G = 0.5 * (self.C + self.D)
        H = 0.5 * (self.D + self.A)
        I = 0.25 * (self.A + self.B + self.C + self.D)

        return [
            Quad(self.A, E, I, H),
            Quad(E, self.B, F, I),
            Quad(I, F, self.C, G),
            Quad(H, I, G, self.D)
        ]

    def integrateOver(self, function):
        result = 0

        for k in range(len(quadQuadPts)):            
            x, y = self.parametricToCartesian(quadQuadPts[k][0], quadQuadPts[k][1])
            result += quadQuadWts[k] * function(x, y)

            plt.plot(x, y, 'o', color = tableauCols(self.Id % 10))

        return self.getArea() * result
    
# ---

# HS

triaQuadPts = [
    np.array([3/5, 1/5, 1/5]),
    np.array([1/5, 3/5, 1/5]),
    np.array([1/5, 1/5, 3/5]),
    np.array([1/3, 1/3, 1/3])
]

triaQuadWts = [
    25/96, 25/96, 25/96, -9/32
]

# ---

class Triangle:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

        self.Id = 0

    def barycentricToCartesian(self, uvw):
        return uvw[0] * self.A + uvw[1] * self.B + uvw[2] * self.C

    def getArea(self):
        return 0.5 * np.linalg.norm(np.cross(self.B - self.A, self.C - self.A)) # Always CCW, i.e. always positive

    def splitLoop(self):
        D = 0.5 * (self.A + self.B)
        E = 0.5 * (self.B + self.C)
        F = 0.5 * (self.C + self.A)

        return [
            Triangle(self.A, D, F),
            Triangle(D, self.B, E),
            Triangle(F, E, self.C),
            Triangle(D, E, F)
        ]
    
    def splitCatMark(self):
        D = 0.5 * (self.A + self.B)
        E = 0.5 * (self.B + self.C)
        F = 0.5 * (self.C + self.A)
        G = (self.A + self.B + self.C) / 3

        return [
            Quad(self.A, D, G, F),
            Quad(D, self.B, E, G),
            Quad(E, self.C, F, G),
        ]        

    def integrateOver(self, function):
        result = 0

        for k in range(len(triaQuadPts)):
            x, y = self.barycentricToCartesian(triaQuadPts[k])
            result += triaQuadWts[k] * function(x, y)

            plt.plot(x, y, 'o', color = tableauCols(self.Id % 10))

        return (self.getArea() / 0.5) * result

# ---

# Use for integrating
unitTriangle = Triangle(np.array([0, 0]), np.array([1, 0]), np.array([0, 1])) 

# Use for visualising
# unitTriangle = Triangle(np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(3) / 2])) 

# ---

def printResult(polyType, numPolys, approxInt, exactInt):
    print("Number of ", polyType, ": ", numPolys, "\tapproxInt: ", approxInt, "\tabs(Error): ", np.abs(exactInt - approxInt), "\tlog(abs(Error)): ", np.log(np.abs(exactInt - approxInt)), sep='')

# ---

function = F3 # When changed, do NOT forget to also update the next line!
exactInt = exactIntF3

approxInt = unitTriangle.integrateOver(function)
printResult("trias", 1, approxInt, exactInt)

triaList = [unitTriangle]
quadList = [unitTriangle] # Yeah, bit of notation abuse here ;)

subdivSteps = 7

# ---

# :: Recursively subdivide the triangle into triangles, integrating over each subtriangle and sum the results
# Use this for G and HS

# for k in range(subdivSteps):
#     # Clear plot (i.e. only show quadrature points of the final subdivision)
#     plt.clf()

#     tempTriaList = []

#     for tria in triaList:
#         tempTriaList += tria.splitLoop()

#     triaList = tempTriaList

#     approxInt = 0
#     triaCount = 0

#     for tria in triaList:
#         tria.Id = triaCount
#         triaCount += 1

#         approxInt += tria.integrateOver(function)

#     printResult("trias", len(triaList), approxInt, exactInt)

# ---

# :: Recursively subdivide the triangle into quadrilaterals, integrating over each quadrilateral and sum the results
# Use this for TP

for k in range(subdivSteps):
    # Clear plot (i.e. only show quadrature points of the final subdivision)
    plt.clf()    

    tempQuadList = []

    # :: Disable for alternative split
    for quad in quadList:
        tempQuadList += quad.splitCatMark()

    # :: Enable for alternative split
    # for tria in triaList:
    #     tempQuadList += tria.splitCatMark()    

    quadList = tempQuadList

    approxInt = 0
    quadCount = 0

    for quad in quadList:
        quad.Id = quadCount
        quadCount += 1

        approxInt += quad.integrateOver(function)

    printResult("quads", len(quadList), approxInt, exactInt)

    # :: Enable for alternative split
    # tempTriaList = []

    # for tria in triaList:
    #     tempTriaList += tria.splitLoop()

    # triaList = tempTriaList

# ---

plt.gca().set_aspect('equal', adjustable='box')
# plt.show()