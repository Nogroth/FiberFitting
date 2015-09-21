from PIL import Image, ImageDraw
from math import sqrt, cos, sin, tan, pi
from scipy import rand
from operator import not_
from EllipseMath import *
from numpy import half
from scipy import optimize


# d is compass direction
def getAheadPoint( x, y, d ):
    if d == 0:
        y += -1
    elif d == 1:
        x += 1
    elif d == 2:
        y += 1
    elif d == 3:
        x += -1
    return x, y

def aheadOk( x, y, d, pixels, w, h, tempColor ):
    
    xp, yp = getAheadPoint(x, y, d)
    #if it's off the board, turn right and try the next point
    if (  (xp < 0)
        | (yp < 0)
        | (xp > w - 1) 
        | (yp > h - 1) ):
        return False
    #now we can use those coordinates because they're on the board
    elif pixels[xp, yp] < tempColor:
        return False
    else:
        return True
    

def getOutline(image, pixels, i, j, width, height):
    ellipsePoints = []
    ellipsePoints.append( (i, j) ) #add the initial point
    
    
    #for each value of c, add all the good points around the c'th point
    #quits when there are no more good points to be added around any existing points
    #meaning all white dots are recorded 
    c = 0
    area = 0
    tempColor = 124
    while c < len(ellipsePoints):
        i = ellipsePoints[c][0]
        j = ellipsePoints[c][1]

        image.putpixel( ellipsePoints[c], tempColor + 1 )
        if (j - 1 >= 0):
            if (pixels[i,j-1] > 127 ):
                image.putpixel( (i, j-1), tempColor )
                ellipsePoints.append( (i, j - 1))
        
        if (i - 1 >= 0):
            if  (pixels[i-1,j] > 127):
                image.putpixel( (i-1, j), tempColor )
                ellipsePoints.append((i - 1, j))
        
        if (i + 1 < width):
            if (pixels[i+1,j] > 127):
                image.putpixel( (i+1, j), tempColor )
                ellipsePoints.append((i + 1, j))
        
        if (j + 1 < height):
            if (pixels[i,j+1] > 127):
                image.putpixel( (i, j+1), tempColor )
                ellipsePoints.append((i, j + 1))
#         print(c)
        c += 1
    
    outline = []
    area = len(ellipsePoints)
    if area < 40:
        return outline, area
    d = 0
    p = (0,0)
    
    
    #find a black pixel and which direction it's in
    for c in range( 0, len(ellipsePoints)):
        i = ellipsePoints[c][0]
        j = ellipsePoints[c][1]

        if (j - 1 >= 0):
            if (pixels[i,j-1] < tempColor ):
                d = 2
                p = (i, j)
                break
        if (i - 1 >= 0):
            if  (pixels[i-1,j] < tempColor ):
                d = 1
                p = (i, j)
                break
        
        if (i + 1 < width):
            if (pixels[i+1,j] < tempColor ):
                d = 3
                p = (i, j)
                break
        
        if (j + 1 < height):
            if (pixels[i,j+1] < tempColor ):
                d = 0
                p = (i, j)
                break
    x0 = p[0]
    y0 = p[1]
    outline.append( (x0, y0) )
    
    '''
    turn right
    while !aheadOk:
        turn left
    move()
    
    d:
    0 = north
    1 = east
    2 = south
    3 = west
    '''
    
#     im2 = im
    while ( True ):
        c = 0
        # turn right
        d += 1
        d %= 4
#         print(pixels[x0,y0])
        while (not aheadOk(x0, y0, d, pixels, width, height, tempColor)):
            c+=1
            #turn left
            d += 3
            d %= 4
            if c > 40:
                print(x0, y0)
                return []
        
        x0, y0 = getAheadPoint(x0, y0, d)
#         print( x0, y0 )
        
        # once you make a full circle, it should always happen
        if ( x0 == outline[0][0]) & ( y0 == outline[0][1] ):
            break
        
        outline.append( (x0, y0) )
#         im2.putpixel( (x0, y0), 80)
#         print(d)
#         im2.show()
#         input("in")
    
#     for c in range(0, len(ellipsePoints)):
#         pointIsGood = False
#         i = ellipsePoints[c][0]
#         j = ellipsePoints[c][1]
#         if (j - 1 >= 0):
#             if (pixels[i,j-1] < tempColor ):
#                 pointIsGood = True
#         
#         if (i - 1 >= 0):
#             if  (pixels[i-1,j] < tempColor):
#                 pointIsGood = True
#         
#         if (i + 1 < width):
#             if (pixels[i+1,j] < tempColor):
#                 pointIsGood = True
#         
#         if (j + 1 < height):
#             if (pixels[i,j+1] < tempColor):
#                 pointIsGood = True
#         
#         if pointIsGood:
#             outline.append( (i, j) )
    
    return outline, area

def getMinWidth( points ):
    half = len(points)/2
    prevDist = 1000
    prevPrevDist = 0
    dist = 0
    minDist = 1000
    for i in range(0, half):
        dist = sqrt( sqrDist( points[i], points[i + half]))
        
        if (dist > prevDist) & (prevDist < prevPrevDist) & (prevDist < minDist):
            minDist = prevDist
        prevPrevDist = prevDist
        prevDist = dist
            
    return minDist

def solve(data, minW):
#     print(data)
    guess = getGuess(data, minW)
    if guess == 0:
        return(0,5)
    a0,b0,h0,k0,t0 = guess[:]
#     print(a0,b0,h0,k0,t0)
    
    p0 = numpy.array([a0,b0,h0,k0,t0])
    
    fitfunc = lambda p, x1, x2: (numpy.power(
                                    ( (x1-p[2])*numpy.cos(p[4]) + (x2-p[3])*numpy.sin(p[4]) ),2
                                            )/numpy.power(p[0],2)
                                 + numpy.power(
                                    ( (x1-p[2])*numpy.sin(p[4]) - (x2-p[3])*numpy.cos(p[4]) ),2
                                            )/numpy.power(p[1],2))
    
    errfunc = lambda p, x, y: fitfunc(p, x, y) - 1 # Distance to the target function
    
    xVec = [ d[0] for d in data ]
    yVec = [ d[1] for d in data ]
    sol = optimize.leastsq(errfunc, p0, args=(xVec,yVec))
    
    # description of other returnable data (like error estimations):
    #     http://stackoverflow.com/questions/24935420/uses-for-secondary-returns-of-scipy-optimize-leastsq
    # example of how to return other data:
    #     http://stackoverflow.com/questions/4520785/how-to-get-rmse-from-scipy-optimize-leastsq-module
    #tutorial for scipy, including some info on optimize.*
    #     http://www.tau.ac.il/~kineret/amit/scipy_tutorial/
    
    # p1, success = scipy.optimize.root(errfunc, p0, args=(xVec,yVec), method = 'lm')

#     a, b, x0, y0, alpha = p1[:]
    return sol

def getMaxDist( points ):
    maxDist = 0
    for i in range(0, len(points)):
        for j in range(0, len(points)):
            
            dist = sqrt( sqrDist( points[i], points[j]))
            
            if (dist > maxDist):
                maxDist = dist
            
    return maxDist

def getApproxMinor( points ):
    prevDist = 1000
    prevPrevDist = 0
    possMinors = []
    minor = 0
    for i in range(0, len(points), 1):
        for j in range(0, len(points), 1):
            sqDist = sqrDist( points[i], points[j])
            
            if ( (sqDist > prevDist) & (prevDist < prevPrevDist) & (prevDist > minor) ):
                possMinors.append( prevDist )
                minor = prevDist
            prevPrevDist = prevDist
            prevDist = sqDist
    for i in range(0, len(possMinors)):
        print(sqrt(possMinors[i]))
    return sqrt(minor)/2

def fillOutline( outline, image, col ):
    draw = ImageDraw.Draw(image)
    draw.polygon(tuple(outline), col)
    
def drawOutline( outline, image, col = 255, shade = False ):
    if shade == False:
        hue = col
    for i in range(0, len(outline)):
        if shade:
            c1 = int(255 * (1- i/len(outline)))
            c2 = int(255 * (i/len(outline)))
            hue = (c1,c2,0)
        try:
            image.putpixel(outline[i], hue)
        except IndexError:
            ()


def readBWPic( minPoints, pixScale, minWidth, maxLength ):
    minW = minWidth / pixScale
    maxL = maxLength / pixScale
    im1 = Image.open( "MatlabStuff/test2.jpg" )
    im = Image.open( "MatlabStuff/test2(bw)(Fix).jpg" )
#     im1 = Image.open( "Images/FiberImages/BOSCH/BOSCH_EGPNylon66_50wt_LGF_2mm_F41_90_W_40_L_3_R(color).jpg" )
#     im = Image.open( "Images/FiberImages/BOSCH/BOSCH_EGPNylon66_50wt_LGF_2mm_F41_90_W_40_L_3_R(color)(bw)(Fix).jpg" )
#     im1 = Image.open( "Images/ellipse.jpg" )
#     im = im1.convert('1')
    pixels = im.load()
#     im.show()
    
    width, height = im.size
    
    out1 = im1
    out2 = Image.new("L", (width, height), 0)
    
    outlines = []
#     go through and find a white(ish) square
    for i in range( 0, width ):
        for j in range( 0, height ):
            if pixels[i,j] > 127:
                outline, area = getOutline(im, pixels, i, j, width, height)
                if area > minPoints:
                    outlines.append( outline )
#                     for c in range( 0, len(outline) ):
#                         p2 = ( outline[c][0], outline[c][1] )
#                         im.putpixel(p2, 100)
#             print(i,width, j, height)
#     im.show()

    start = 0
    end = len(outlines)

# 123 is noise, 131 is lovely, 143 is a square block
#     start = 110
#     end = 123
    for i in range(start, end):
#         string = str(i) + " " + str(float(100*(i - start))/(end - start)) + "%"
#         print(string)
        list1 = outlines[i]
        
#         noisiness = getCurvature( list1 )
#         
# #         print( noisiness )
#         for j in range( 0, len(list1) ):
#             if noisiness > 0.6:
#                 col = (255,0,0)
#             else:
#                 col = (0,255,0)
#             
# #             c = float(255*(noisiness))
# #             col = (int(255 - c), int(c), int(255 - c))
#             out1.putpixel(list1[j], col)
#             
#         continue
#        #manually found good numbers for 131
#         a, b, h, k, t = 55.5, 19, 410, 1065.5, 1.65
#        #the 'best' calculated numbers for 123
#         a, b, h, k, t = 29.71, 3.387, 376.2, 545.3, 0.88024544
#         a, b, h, k, t, junk, junk2 = getBestFit(list1, minW, maxL)
#         print( a, b, h, k, t)
#         x = 367
#         y = 1430
#         scale = ((x - h)*cos(t) + (y - k)*sin(t))**2 / a**2 + ((x - h)*sin(t) - (y - k)*cos(t))**2 / b**2
#         print(abs(scale - 1), " scale")
#         print( closeness3(list1, a, b, h, k, t), "closeness" )
#         
#         result = getEllipseOutline(h, k, t, a, b, True)
#         out1.putpixel((x,y), (0,0,255))
#         for j in range( 0, len(result)):
#             try:
#                 out1.putpixel(result[j], (0,255,0))
#             except IndexError:
#                 ()
#              
#         for j in range( 0, len(list1) ):
#             out1.putpixel(list1[j], (255,0,0))
#         out1.show()
# #         input("xdfgs")
#         break
#         avg = getMinWidth(list1)
#         print(avg,1)
#         aTemp = getMaxDist( list1 )
#         print(aTemp,2)
#         bestI = 0
#         bestpercent = 0

#         a, b, h, k, t, percentSame, points = getBestFit(list1, minW, maxL)
        sol = solve(list1, minW/2)
        if sol[1] > 4:
            continue
        a, b, h, k, t = sol[0][:]
#         print(i, (a,b,h,k,t))
        if ( a > width) | (b > 3*minW):
            continue
        result = getEllipseOutline(h, k, t, a, b, True)
        fillOutline(result, out2, 255)
        
        print(i+1, "of", end-start)
#         a *= 1.04
#         b *= 1.04
#         print(percentSame)
#         print("")
#         if percentSame < 0.30:
#             continue
#         for i in range( 0, len(list1) ):
#             p1 = ( list1[i][0], list1[i][1] )
#             out1.putpixel(p1, (255,0,0))
#         fillEllipse(out, h, k, t, a, b)
#         for i in range( 0, len(result) ):
#             p2 = ( result[i][0], result[i][1] )
# #             print(p2)
#             try:
#                 out1.putpixel(p2, 90)
#             except IndexError:
#                 ()
#         for i in range( 0, len(list1) ):
#             out1.putpixel(list1[i], (0,255,0))
    #     im1.show()
#         break
#     out1.show()
    out2.save("firstOutput.bmp")
    
    return


# def func(coeffs, v):
# #     print(coeffs, v)
#     a,b,c,d,e,f = coeffs[:]
# #     return "b"
#     x,y = v[:]
#     return [ a*x**2, b*x*y, c*y**2, d*x, e*y, f*1 ]

# def func(v):
#     x,y = v[:]
#     return [ x**2, x*y, y**2, x, y, 1 ]
# 
# def jacob(coeffs, v):
# #     print(coeffs, v)
#     a,b,c,d,e,f = coeffs[:]
#     x,y = v[:]
#     return numpy.array([[2*a*x, 0],
#                        [b*y, b*x],
#                        [0, 2*c*y],
#                        [d, 0],
#                        [0, e],
#                        [0, 0]])
# #     return numpy.array([[2 * x, y, 0, 1, 0, 0],
# #                         [0, x, 2*y, 0, 1, 0]])
# 
# 
# def printStuff(a,b,c,d,e):
#     print('{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}'.format(a, b, c, d, e) )

# a = 93
# b = 157
# h = 150
# k = 200
# t = 1.3
# c = cos(t)
# s = sin(t)
# data = getEllipseOutline(h, k, t, a, b, True)
#  
# A1 = (b*c)**2 + (a*s)**2
# B1 = -2*c*s*(a**2 - b**2)
# C1 = (b*s)**2 + (a*c)**2
# D1 = -2*A1*h - k*B1
# E1 = -2*C1*k - h*B1
# F1 = -1*(a*b)**2 + A1*h**2 + B1*h*k + C1*k**2
# A1 /= -F1
# B1 /= -F1
# C1 /= -F1
# D1 /= -F1
# E1 /= -F1
  
 
# A0, B0, C0, D0, E0 = generalizedSolve( getSpacedPoints(data, 0, int(0.6*len(data))) )
# x0 = [A0, B0, C0, D0, E0, 1]
# x0 = [2,2]
# x0 = [A1, B1, C1, D1, E1, 1]

# import scipy, scipy.optimize, scipy.io
# 
# fitfunc = lambda p, x1, x2: p[0]*numpy.power(x1,2) + p[1]*x1*x2 + p[2]*numpy.power(x2,2) + p[3]*x1 + p[4]*x2 + p[5]
# 
# errfunc = lambda p, x, y: fitfunc(p, x[0], x[1]) - y # Distance to the target function
# 
# data = numpy.array([(4, 0), (3, 0), (3, 1), (0, 1), (-1, 1), (-2, 1), (-4, 0), (-2, -1), (0, -1), (2, -1), (3, -1)])
# x0 = numpy.array([0.0625, -0.0, 0.25, 0.0, 0.0, 1])
# xVec = [ d[0] for d in data ]
# yVec = [ d[1] for d in data ]
# p1, success = scipy.optimize.leastsq(errfunc, x0, args=(xVec,yVec))
# print(p1)
# 
# 
# print(1/0)
# 
# g = numpy.linspace(0., 8., 4)
# print(numpy.power(g,2))
# num_points = 5
# Tx = numpy.linspace(0., 8., num_points)
# Tx = [ [X,0] for X in Tx ]
# tX = [ list(numpy.power(X,2)) for X in Tx ]
# print(Tx)
# print(tX)
# fitfunc = lambda p, x, x2: float(p[0])*numpy.cos(x) + x # Target function
# errfunc = lambda p, x, y: fitfunc(p, x,x[1]) - y # Distance to the target function
# p0 = [-15., 0.8, 0., -1.] # Initial guess for the parameters
# print( optimize.leastsq(errfunc, p0[:], args=(Tx, tX))[0] )
# 
# def fit2(A,b):
#     """ Relative error minimizer """
#     def f(x):
#         assert len(x) == len(A[0])
#         resids = []
#         for i in range(len(A)):
#             sum = 0.0
#             for j in range(len(A[0])):
#                 sum += A[i][j]*x[j]
#             relative_error = (sum-b[i])/b[i]
#             resids.append(relative_error)
#         return resids
#     ans = scipy.optimize.leastsq(f,[0.0]*len(A[0]))
# 
# print(1/0)
# 
# data = numpy.array([(4, 0), (3, 0), (3, 1), (0, 1), (-1, 1), (-2, 1), (-4, 0), (-2, -1), (0, -1), (2, -1), (3, -1)])
# x0 = numpy.array([0.0625, -0.0, 0.25, 0.0, 0.0, 1])
# 
# fitfunc = lambda p, x1, x2: p[0]*numpy.power(x1,2) + p[1]*x1*x2 + p[2]*numpy.power(x2,2) + p[3]*x1 + p[4]*x2 + p[5]
# 
# errfunc = lambda p, x, y: fitfunc(p, x[0], x[1]) - y # Distance to the target function
# 
# xVec = [ d[0] for d in data ]
# yVec = [ d[1] for d in data ]
# sol = optimize.leastsq(errfunc, x0[:], args=(xVec, yVec))
# 
# # sol = optimize.root(fun=func, x0=x0, args=data, jac=jacob, method='lm' )
# 
# # sol = optimize.fsolve(func, data)
# print( sol)
# print( sol.x )
# [A, B, C, D, E] = sol.x
# printStuff(A1, B1, C1, D1, E1)
# printStuff(A0, B0, C0, D0, E0)
# printStuff(A, B, C, D, E)
# print("")
#   
# a1, b1, h1, k1, t1 = getParameters(A1,B1,C1,D1,E1, 20)
# a0, b0, h0, k0, t0 = getParameters(A0,B0,C0,D0,E0, 20)
# a, b, h, k, t = getParameters(A,B,C,D,E, 20)
#   
# printStuff(a1, b1, h1, k1, t1)
# printStuff(a0, b0, h0, k0, t0)
# printStuff(a, b, h, k, t)


'''
''

def fun(x):
    return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
            0.5 * (x[1] - x[0])**3 + x[1]]


def jac(x):
    return numpy.array([[1 + 1.5 * (x[0] - x[1])**2,
                     -1.5 * (x[0] - x[1])**2],
                     [-1.5 * (x[1] - x[0])**2,
                     1 + 1.5 * (x[1] - x[0])**2]])

sol = optimize.root(fun=fun, x0=[0,0], jac=jac, method='lm')
print(sol.x)



''
'''

# from scipy.optimize import curve_fit
# 
# xdata = numpy.array([-2,-1.64,-1.33,-0.7,0,0.45,1.2,1.64,2.32,2.9])
# ydata = numpy.array([0.699369,0.700462,0.695354,1.03905,1.97389,2.41143,1.91091,0.919576,-0.730975,-1.42001])
# 
# def func3(x, p1,p2):
#   return p1*numpy.cos(p2*x) + p2*numpy.sin(p1*x)
# 
# x = curve_fit(func3, xdata, ydata,p0=(1.0,0.2))
# print(x)


readBWPic(300, 1, 21, 185)



# im = Image.new("L", (400, 400), 0)
# fillEllipse(im, 200, 200, 2, 80, 79)
# outline = getEllipseOutline(200, 200, 2, 80, 79, False)
# 
# localSame = closeness2(outline, 80, 79, 200, 200, 2, im)
# print( localSame )
# im.show()



# im = Image.open( "Images/ellipse.jpg" )
# im = im.convert('1')
# pixels = im.load()
#     
# width, height = im.size
#     
# outline = ()
# 
# 
# outline = getOutline(im, pixels, 110, 83, width, height)
# 
# for i in range( len(outline)):
# #     print(i)
#     im.putpixel( outline[i], 255)
# im.show()

# def printTestEllipse():
#     a = 200
#     b = 20
#     h = 500
#     k = 411
#     t = 1.3
#     c = cos(t)
#     s = sin(t)
#       
#     A = (b*c)**2 + (a*s)**2
#     B = -2*c*s*(a**2 - b**2)
#     C = (b*s)**2 + (a*c)**2
#     D = -2*A*h - k*B
#     E = -2*C*k - h*B
#     F =-1*(a*b)**2 + A*h**2 + B*h*k + C*k**2
#      
#     print(A,B,C,D,E,F)
#      
#     print(getParameters(A, B, C, D, E, F))
#     list1 = getEllipseOutline(h, k, t, a, b)
# 
#     for i in range(45, len(outlines)):
#            
#         list1 = outlines[i]
#         im = im1
#         for j in range( 0, len(list1) ):
#             p1 = ( list1[j][0], list1[j][1] )
#             im.putpixel(p1, 255)
#         print(i)
#         im.show()

