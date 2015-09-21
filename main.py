'''

Christopher Hill


Apparently someone else has thought up the rotational binary search before
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.308.1650&rep=rep1&type=pdf
It's a description of finding skew in an image, by finding long lines which have been skewed, 
    and doing a "interval-halving binary search" to find the angle at which the maximum
    number of lines are in the same direction as that angle.
    Everything but the binary search is different though, and it doesn't solve my bug.
    
    Actually it mentions estimating a confidence value in its binary thing by measuring
    both the min and the max in that binary search region, and returning max/min.
    If max/min ~= 1, then it's not good.

'''

from PIL import Image, ImageDraw
import math
from colorSpread import getStats
from EllipseMath import sqrDist
from numpy.linalg import *
from numpy import *
import numpy
from scipy import optimize
from audioop import avg
import EllipseMath
from scipy.constants.constants import foot


def withinTol( color1, color2, tol ):
    r = abs(color1[0] - color2[0])
    g = abs(color1[1] - color2[1])
    b = abs(color1[2] - color2[2])
    #print "r:" + str(r) + " g:"+ str(g) + " b:"+ str(b)
    return (r < tol[0]) & (g < tol[1]) & (b < tol[2])

#takes input of a coordinate (in (x,y) format, relative to the origin) on the regular xy plane
#gives theta as measured from the x axis. (polar)
def getTheta(coord):
    theta = math.atan( float(coord[1]) / (coord[0]+0.000001) )
#     print(theta)
    if coord[0] < 0:
        theta += math.pi
    elif coord[1] < 0:
        theta += math.pi * 2
    return theta


#function that will calculate the amount that side2 of a given 9x9 square is lighter than side1, 
#based on an input rotation of the centerline. the centerline starts on the x-axis.
#side1 is the centered at theta + pi/2

def getLightnessDiffference( square, theta ):
    theta -= 0.0001
    side1 = [0,0,0]
    side2 = [0,0,0]
    boxW = len(square[0])
    lBound = int(0 - boxW/2)
    hBound = boxW + lBound #ensures odd boxW's don't result in too short a range due to integer division
    s1Count = 0
    s2Count = 0
    
    for i in range(lBound, hBound):
        for j in range(lBound, hBound):
#                 print(i, j, k)
            tempTheta = getTheta( (i, j) )
            lThetaBnd = theta
            hThetaBnd = theta + math.pi
#             print(tempTheta, lThetaBnd, hThetaBnd, lThetaBnd - 2* math.pi, hThetaBnd - 2* math.pi )
            while (tempTheta < lThetaBnd) & (hThetaBnd > 2* math.pi):
                hThetaBnd -= 2* math.pi
                lThetaBnd -= 2* math.pi
            while (tempTheta > hThetaBnd) & (lThetaBnd < 0):
                hThetaBnd += 2* math.pi
                lThetaBnd += 2* math.pi
                
            if (tempTheta > lThetaBnd) & ( tempTheta < hThetaBnd):
#                     im.putpixel((i+x,j+y), (0,255,0))
                for k in range(0,3):
                    side1[k] += square[i][j][k]
                s1Count += 1
            else:
#                     im.putpixel((i+x,j+y), (255,0,0))
                for k in range(0,3):
                    side2[k] += square[i][j][k]
                s2Count += 1
    difference = [0,0,0]
    for i in range(0,3):
#         print(side1[i]/float(s1Count))
        difference[i] = side1[i]/s1Count - side2[i]/s2Count
#     print(int(theta/math.pi * 180+0.5), s1Count, s2Count)
#     im.save("temp.bmp")
    return difference
  
def getSquare( pixels, boxW, x, y ):
    square = []
    lBound = int(0 - boxW/2)
    hBound = boxW + lBound
    for i in range(lBound, hBound):
        square.append([])
        for j in range(lBound, hBound):
            square[len(square)-1].append( pixels[i + x, j + y] )
    return square

#returns if a1 is bigger on average than a2
def biggerThan( a1, a2, tol = (0,0,0) ):
    sumDiffs = 0
    for i in range(0, len(a1)):
        sumDiffs += (a1[i] - a2[i] - tol[i])
    return sumDiffs > 0

#returns if the absVal of a1 is bigger than the absVal of a2
def absBiggerThan( a1, a2 ):
    sumDiffs = 0
    for i in range(0, len(a1)):
        sumDiffs += (abs(a1[i]) - abs(a2[i]))
    return sumDiffs > 0

def sqrDist( p1, p2 ):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# def getSplotchyImage( im, cutoff ):
#     pixels = im.load()
#     width, height = im.size
#     for i in range( 0, width):
#         for j in range( 0, height):
#             
#             if pixels[i,j]
def getMidPoint(p1, p2):
    return ( int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2) )

def getLineMatrix( p1, p2 ):
    m = (p2[1] - p1[1]) / (p2[0] - p1[0]+0.00001)
    b = p1[1] - m * p1[0]
    line = []
    line.append(p1)
    for x in range(p1[0]+1, p2[0]):
        p = (x, int(m * x + b))
        if p != line[len(line)-1]:
            line.append( p )
    return line

# def drawLineMatrix( pix, p1, p2 ):
#     
#     line = getLineMatrix(pix, p1, p2)
#     
#     
# def getPolygonMatrix( pix, points ):
#     outline = []
#     for i in range(0, len(points)-1):
#         outline += getLineMatrix(pix, points[i], points[i+1])
#     outline += drawLineMatrix(pix, points[len(points)-1], points[0])
#     return outline
# 
# 
# def drawPolygonMatrix( pix, points ):
#     outline = []
#     for i in range(0, len(points)-1):
#         outline += drawLineMatrix(pix, points[i], points[i+1])
#     drawLineMatrix(pix, points[len(points)-1], points[i+1])
# 
#     
# def fillPolygonMatrix( pix, points ):
#     assert False == True
#     drawPolygonMatrix(pix, points)
#     

def getOutline(pixels, w, h, boxW, maxLength, i, j, avg, tol):
    '''
    input is the first boundary point.
    '''
    skipSize = 1.7 * sqrt(2)
    outline = []
    prevP = (i, j)
    outline.append(prevP)
#     footprints = [ [0]*h ] * w
    footprints = numpy.zeros(shape=(w,h))

    sqr = getSquare(pixels, boxW, i, j)
    t = getBestAngle(sqr)

    cutOffSqrDist = (3*skipSize)**2

    while True:
        if len(outline) > 15:
            dist = sqrDist(prevP, outline[0])

            if dist <= cutOffSqrDist:
                print("close to start")
                break
            elif dist > maxLength**2:
                print("too far away")
                break
            if len(outline) > (maxLength / skipSize)*3:
                print("too many points")
                break

        i += skipSize * math.cos(t)
        j += skipSize * math.sin(t)

        nxtP, t, diff = getBestInRegion(pixels, w, h, boxW, i, j, 2, skipSize, prevP, 'n', avg, tol, len(outline))
        i,j = nxtP

        nxtP = tuple([int(x) for x in nxtP])

        if nxtP == (0,0):
            print("No valid next point", prevP, len(outline))
            outline = []
            break
#         print(nxtP, prevP)
#         print(nxtP, prevP,footprints[nxtP[0]][nxtP[1]], footprints[prevP[0]][prevP[1]])
        if nxtP != prevP:
#             index = footprints[nxtP[0]][nxtP[1]]
#             if index != 0:
#                 # means that that point has been printed before; it could have been printing points
#                 # right next to previous points though, before it hit this one right on.
#                 # so backtrack until the examined point is far enough from any other points, and
#                 # then re-define the list to cut off everything afterwards - 1.
            looping = False
            line = getLineMatrix( prevP, nxtP)
            for p in line:
#                 print(p)
                if footprints[p[0]][p[1]] != 0:
                    looping = True
            if looping:
                # means that nxtP and prevP are on either side of a previously drawn line; means that that section
                # has been traced before, and so the loop needs to stop.
                start = 0
                for start in range(len(outline)-4, -1, -1):
#                     print(start, sqrDist(outline[start], outline[len(outline)-1]), cutOffSqrDist / 5)
                    if sqrDist(outline[start], outline[len(outline)-1]) < cutOffSqrDist / 5:
                        break
                outline = outline[start:]
                
                print("Went into loop")
#                 print(1/0)
                break
                
            if ((i - boxW/2 < 0)
                |(j - boxW/2 < 0)
                |(i + boxW/2 > w)
                |(j + boxW/2 > h)):
#                 print(i,j, t, prevT, nxtP)
                print("Went off-screen")
                break
            
            outline.append(nxtP)
#             footprints[nxtP[0]][nxtP[1]] = len(outline) #this marks footprints for each point
            for p in line: # this marks footprints both at and between each point
                footprints[p[0]][p[1]] = len(outline)
            prevP = nxtP

    return outline

#this returns an angle measured from the x-axis. White is pi below this angle, black is pi above.
#it works analogously to a binary search. If the value it calculates is negative, it turns one way.
# if it's positive, it turns the other. The amount it turns is half of the amount it turned the previous time.
# it tries to find the angle that splits the amount of color equally on each side, and returns that angle + pi/2.
# It assumes that the angle of no contrast is 90 degrees off of the angle of highest contrast. I think it's true
# by definition. 
def getBestAngle(sqr):
    
#     original loop, before being shortened. Saved for reference if necessary
#     measuredAngle = 0
#     #the first use of it should be 90, so start it at 180
#     halfPrevMovement = 180
#     
#     #test to find out how many iterations are necessary
#     for i in range(0, 7):
#         usedAngle = measuredAngle - 90
#         usedTheta = usedAngle * math.pi / 180
#         diff = getLightnessDiffference(sqr, usedTheta)
#         halfPrevMovement /= 2.0
#         if biggerThan(diff, (0,0,0)):
#             measuredAngle -= halfPrevMovement
#         else:
#             measuredAngle += halfPrevMovement
#         print(i, (measuredAngle - 90) * math.pi / 180)
#     
    halfPrevMovement = math.pi
    usedTheta = - math.pi / 2
    for i in range(0, 8):
        diff = getLightnessDiffference2(sqr, usedTheta)
        halfPrevMovement /= 2.0
        if biggerThan(diff, (0,0,0)):
            usedTheta -= halfPrevMovement
        else:
            usedTheta += halfPrevMovement
        i+=0
#         print(i, usedTheta)
    return usedTheta + math.pi / 2

def checkPoint( square ):
    
    theta = getBestAngle(square)
    diff = getLightnessDiffference2(square, theta)
    
    return theta, diff


def getBestInRegion( pixels, width, height, boxW, x, y, searchSize, skipSize, prevP, prevT, avg, tol, tempCount=0 ):
    bestDiff = [0,0,0]
    bestT = 0
    bestP = (0,0)
    r = searchSize # x pixels away from the starting pixels, inclusive
    for j in range(-r, r+1, 1 + int(r/3)):
        for i in range(-r, r+1, 1 + int(r/3)):
            
#             print(i,)
            if x + i + 4 >= width:
                break
            elif y + j + 4 >= height:
                break
            elif x + i - 4 < 0:
                i+= 4
            elif y + j - 4 < 0:
                j+= 4
            else:
                pointIsGood = False
                if prevP != 'n':
                    if sqrDist(prevP, (x + i, y + j)) > skipSize**2:
                        midP = getMidPoint(prevP, (x + i, y + j))
                        # this checks if the difference is bigger than the tolerance
                        # but it's only true if the pixels are more than tol darker than the average,
                        # not tol lighter.

                        diff = diffVec(avg, pixels[midP[:]])
                        
                        if not biggerThan(  diff, tol):
                            pointIsGood = True
#                         else:
#                             print(tempCount, prevP, midP,(x + i, y + j),"midpoint too dark")
                else:
                    pointIsGood = True

                #if it's on the screen
                if pointIsGood & (x + i < width - boxW/2) & (y + j < height - boxW/2):

                    square = getSquare(pixels, boxW, x + i, y + j)
                    
                    theta, diff = checkPoint(square)

                    if absBiggerThan(diff, bestDiff):
                        bestDiff = diff
                        bestT = theta
                        bestP = (x + i, y + j)

    return bestP, bestT, bestDiff
        

#alpha is angle that the ellipse has been rotated
#equation gotten from
# http://math.stackexchange.com/questions/426150
def drawEllipse(image, x0, y0, alpha, a, b, col = (255,255,255)):
    for t in range(0, 720):
        theta = t * math.pi/360
        x = x0 + a * math.cos(theta) * math.cos(alpha) - b * math.sin(theta) * math.sin(alpha)
        y = y0 + a * math.cos(theta) * math.sin(alpha) + b * math.sin(theta) * math.cos(alpha)
        try:
            image.putpixel( (int(x), int(y)) , col)
        except Exception:
            ()
    
def getEllipseOutline( x0, y0, alpha, a, b, fine):
    outline = []
    x = x0 + a * math.cos(0) * math.cos(alpha) - b * math.sin(0) * math.sin(alpha)
    y = y0 + a * math.cos(0) * math.sin(alpha) + b * math.sin(0) * math.cos(alpha)
    outline.append( (int(x), int(y)) )
    
    numPoints = int(2 * math.pi * math.sqrt( (a**2 + b**2) / 2))
    numPoints = numPoints if fine else 720
    for t in range(1, numPoints):
        theta = t * 2 * math.pi / numPoints
        x = x0 + a * math.cos(theta) * math.cos(alpha) - b * math.sin(theta) * math.sin(alpha)
        y = y0 + a * math.cos(theta) * math.sin(alpha) + b * math.sin(theta) * math.cos(alpha)
        if (int(x), int(y)) != outline[len(outline)-1]:
            outline.append( (int(x), int(y)) )
    return outline

def fillEllipse( im, x0, y0, alpha, a, b, col):
#     '''
#     assume points are in order
#     draw a line from each point to the next
#     find an interior point
#     fill the area using magical black box
#     '''
    pnts = getEllipseOutline(x0, y0, alpha, a, b, True)
    draw = ImageDraw.Draw(im)
    draw.polygon(pnts, fill=col)
#     
#     s = len(pnts)
#     for i in range(0, s):
#         draw.line( ( int(pnts[i][0]), int(pnts[i][1]), int(pnts[(i + 1) % s][0]), int(pnts[(i + 1) % s][1]) ), col, 1)
#     
#     insidePnt = (int(pnts[0][0] + (pnts[int(s/2)][0] - pnts[0][0]) / 2), int(pnts[0][1] + (pnts[int(s/2)][1] - pnts[0][1]) / 2))
#     
#     ImageDraw.floodfill(im, insidePnt, col, border=None)

def roundToNearest(x):
    return int(round(x) - .5) + (x > 0)

def sumVec(v1,v2):
    return (v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2])

def diffVec(v1,v2):
    v = (v1[0]-v2[0], v1[1]-v2[1], v1[2]-v2[2])
#     print(v, v1[0], v2[0])
    return v


def vecMag(v):
    return sqrt(v[0]**2 + v[1]**2)

def getVecInOutline(i1, i2, outline):
    return (outline[ i1 ][0]-outline[ i2 ][0], outline[ i1 ][1]-outline[ i2 ][1])

def getAngle( v1, v2 ):
    denom = vecMag(v1)*vecMag(v2)
    cosAngle = dot(v2,v1)/denom
    return cosAngle
    
def isBreakPoint(outline, i):
    
    '''
    make it take into account the sine of the angle (cross product?), and from that deduce the quadrant that the angle
    is in, and thereby be able to distinguish 179 degrees from 1 degree.
    '''
    i3a = (i-3) % len(outline)
    i2a = (i-2) % len(outline)
    i1a = (i-1) % len(outline)
    i0 = i
    i1b = (i+1) % len(outline)
    i2b = (i+2) % len(outline)
    i3b = (i+3) % len(outline)
#     v1 = numpy.array([[outline[ i0 ][0]-outline[ i1 ][0]], [outline[ i0 ][1]-outline[ i1 ][1]]])
#     v2 = numpy.array([[outline[ i0 ][0]-outline[ i2 ][0]], [outline[ i0 ][1]-outline[ i2 ][1]]])

    a1 = getAngle(getVecInOutline( i0, i1a, outline), getVecInOutline( i0, i1b, outline))
    a2 = getAngle(getVecInOutline( i0, i2a, outline), getVecInOutline( i0, i2b, outline))
    a3 = getAngle(getVecInOutline( i0, i3a, outline), getVecInOutline( i0, i3b, outline))
    
    angle = (a1 + a2 + a3)/3
    
#     sinAngle = cross(v2,v1)/denom
#     print(cosAngle, outline[i1], outline[i0], outline[i2], cosAngle > -0.0001)
    return angle > -1/2


#returns the chunk of the list between those two indices, wraps around if necessary
def getSlice(outline, i1,i2):
    i1 %= len(outline)
    i2 %= len(outline)
    if i2 > i1:
        return outline[i1:i2]
    else:
        return outline[i1:len(outline)] + outline[0:i2]
    
'''
if a point has too high of a curvature, replace that element of splitlist with the first part of that section,
and add the other to the end. If either section is too small, don't add it.
'''
def splitOutline( outline ):
#     outline = [
#                (14,5),
#                (10,8),
#                (9,12),
#                (11,17),
#                (14,21),
#                (10,23),
#                (8,26),
#                (7,30),
#                (8,35),
#                (16,36),
#                (23,36),
#                (24,32),
#                (23,28),
#                (22,24),
#                (26,24),
#                (28,20),
#                (26,14),
#                (23,10),
#                (20,6)
#                ]
    
    splitList = []

#     splitList.append(outline)
#     return splitList

    start = -1
    for i1 in range(0, len(outline)):
        if isBreakPoint(outline, i1):
            start = i1
            break
    if start == -1:
        splitList.append(outline)
        return splitList
    
    bP = [] #break points
    print("Start:",start)
    for i2 in range(0, len(outline)):
        if isBreakPoint(outline, (i2 + start)%len(outline)):
            bP.append((i2 + start)%len(outline))
    print("BP: ", bP)
    for i3 in range(0, len(bP)):
        if abs(bP[i3]+1 - bP[(i3-1)%len(bP)]) > 0:
#             splitList.append(outline[bP[i-1]:bP[i]])
            splitList.append(getSlice(outline, bP[(i3-1)%len(bP)], bP[i3]+1))
    print("SplitList:", splitList)
#     im = Image.new('RGB',(500,600), (0,0,0))
#     
#     for i in range(0, len(splitList)):
#         for j in range(0, len(splitList[i])):
#             p = (splitList[i][j][0],splitList[i][j][1])
#             c = int(i/len(splitList) * 255)
#             c = i%2 * 255
#             col = (c,0,255-c)
#             im.putpixel(p, col)
#     im.show()

    print("SplitList", len(splitList), len(splitList[int(len(splitList)/2)]))
    return splitList

def compareEllipses(e1, e2):
    
    return True

def avgEll( e1, e2 ):
    a = (e2[0] + e1[0])/2
    b = (e2[1] + e1[1])/2
    h = (e2[2] + e1[2])/2
    k = (e2[3] + e1[3])/2
    t = (e2[4] + e1[4])/2
    return ( a, b, h, k, t ) 

def sqrOk( square, avg, tol ):
    #returns true if the square is worth checking for an angle
    #checks if at elast one set of opposite corners are different enough from each other.
#     diag1good = absBiggerThan( diffVec(square[0][0], square[len(square)-1][len(square)-1]), tol)
#     diag2good = absBiggerThan( diffVec(square[0][len(square)-1], square[len(square)-1][0]), tol)
# 
#     return diag1good | diag2good
    # this works the same but is theoretically a little faster
    if absBiggerThan( diffVec(square[0][0], square[len(square)-1][len(square)-1]), tol) == False:
        return absBiggerThan( diffVec(square[0][len(square)-1], square[len(square)-1][0]), tol)
    return True

'''

maybe:

put it through a really really simple threshhold filter thingy-
    everything above avg + n*stdev = white
    everything below avg + n*stdev = black
But who knows if that'd work. If it did, I'd just put a point in the center of each of those splotches,
and have it work just like the above one. The only issue is that it would detect light colored things in
the background (esp the blue image) and try to calculate it for those.

Fix the thing so instead of getting 10 backwards things, give it a certain bank of distance backwards
for it to use. Or maybe just a distance backwards maximum cutoff for each point. How far (distance, not angle)
 to the right is the point from the line made by the two previous points. ?
 ignore to the left? Or just do the thing and if the jump is too big in either direction, split the ellipse.
 No. Find a way to check the curvature of the last x points and split it based on that.
 
 Maybe try that parametric thingy online. Maybe just see if you can make approximate a curve with x points using 
 something like
 x = at^3 + bt^2 + ct + d
 y = et^3 + ft^2 + gt + h
 and solve it using the last 4 points? 4 points relatively equally spaced over the last x pixels of distance?
 Like go backwards in the list until you find a point more than x distance away, and then use 4 spaced points
 from that sublist. 
 
 TODO
 approximate curvature of last x points, for splitting, and compare those two generated ellipses to each other
 find good initial points - coarse filter, take middle point(?)
 measure closeness of ellipse - matrixy stuff
 try other approximation methods
 
'''

def printOutline( outline, image, col ):
    draw = ImageDraw.Draw(image)
    
    s = len(outline)
    for i in range(0, s):
        draw.line( ( int(outline[i][0]), int(outline[i][1]), 
                     int(outline[(i + 1) % s][0]), int(outline[(i + 1) % s][1]) ), 
                   col, 1)

def fillOutline( outline, image, col ):
    draw = ImageDraw.Draw(image)
    draw.polygon(tuple(outline), col)
    
def drawOutline( outline, image):
    for i in range(0, len(outline)):
        c1 = int(255 * (1- i/len(outline)))
        c2 = int(255 * (i/len(outline)))
        hue = (c1,c2,0)
        image.putpixel(outline[i], hue)

def solve(data, minW):
    try:
        a0,b0,h0,k0,t0 = EllipseMath.getGuess(data, minW)
        
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
        p1, success = optimize.leastsq(errfunc, p0, args=(xVec,yVec))
        # p1, success = scipy.optimize.root(errfunc, p0, args=(xVec,yVec), method = 'lm')
        a,b,h,k,t = p1[:]
        if b > a:
            t += pi/2
            temp = a
            a = b
            b = temp
        return a,b,h,k,t
    except Exception:
        print("died", Exception.args)
        return [0,0,0,0,0]

def getRect(pixels, x, y, w, h):
    return pixels[x:x+w,y:y+h]

def getRegionalStats(box):
    colorCount = [0,0,0]
    boxes = 0
    
    for i in range(0,len(box)):
        for j in range(0,len(box[i])):
            for c in range(0,3):
                colorCount[c] += box[i][j][c]
            boxes += 1
    
    for i in range(0, 3):
        colorCount[i] /= boxes
    
    stdev=[]
    
    for c in range(0,3):
        stdev.append(0)
        for i in range(0,len(box)):
            for j in range(0,len(box[i])):
                stdev[c] += (box[i][j][c] - colorCount[c])**2
        stdev[c] = sqrt(stdev[c] / (len(box)*len(box[i])) )
    return stdev, colorCount


def standAlone( imName ):

#     im = Image.open("Images/rods.jpg")
#     im = Image.new("RGB", (100, 100), (40,80,160))
#     im = Image.open("Images/testPict.jpg")
#     im = Image.open("Images/testPictTiny.jpg")
#     im = Image.open("Images/testPict2.jpg")
#     im = Image.open("Images/test2.jpg")
#     im = Image.open("SplitTest.bm")
#     im = Image.open("Images/ellipse.jpg")
#     im = Image.open("Images/ellipseBlurred.jpg")
#     im = Image.open("Images/tinyTest.jpg")
#     im = Image.open("Images/FiberImages/BOSCH/BOSCH_EGPNylon66_50wt_LGF_2mm_F41_90_W_90_L_3_R(color).jpg")
#     im.show()
    im = Image.open(imName)
    im2 = im.copy()

    boxW = 7 # width of the square used to find boundaries
      
    minWidth = 20
    maxLength = 40*minWidth
      
    width, height = im.size
    out = Image.new("RGB", (width, height), (0,0,0))
    pixels = im.load()
    
    print("Image loaded.")
    
    print("Getting statistics...")
    stdev, avg = getStats(pixels, width, height)
    
    #the tolerances for picking a "good" point
#     highContrast = [ 3*x/8 for x in stdev ]
#     midContrast = [ x/4 for x in stdev ]
#     lowContrast = [ x/8 for x in stdev ]
    highContrast = [ 2*x/8 for x in stdev ]
    midContrast = [ 5*x/32 for x in stdev ]
    lowContrast = [ x/8 for x in stdev ]
    
    fillCol = avg
    for i in range(0, len(fillCol)):
        fillCol[i] = int(fillCol[i] - 2*highContrast[i])
    fillCol = tuple(fillCol)
    
    print("avg = ", avg)
    print("stdev = ", stdev)
    
    #LATER - increase the iterator maybe, once the ellipse finding works.
    # it probably could be much higher than it is now. This loop only needs to find 1 dot
    # on every ellipse; not all of the dots.
    # 14 missed an ellipse
    skipSize = int(boxW / 2)

    outlineList = []

    # center image on dark background
    background = Image.new('RGB', (width + 2*boxW, height + 2*boxW), fillCol)
    bg_w, bg_h = background.size
    offset = (int((bg_w - width) / 2), int((bg_h - height) / 2))
    background.paste(im2, offset)
    # re-load these variables
    im2 = background
    im = im2.copy()
    width,height = im2.size
    pixels = im2.load()

    startX = offset[0]
    startY = offset[1]
#     startX = 280
#     startY = 202
#     startX = 28
#     startY = 150
    
    for j in range(startY, height - offset[1], skipSize):
#         if len(outlineList)>0:
#             break
        for i in range(startX, width - offset[0], skipSize):

            '''
            1. check point
            2. if point is good, try to find an ellipse
            3. if there are enough points for an ellipse, add it to the list
            '''
            #test on 401x302
            #took about 23 mins, most of which felt like the continuous loops
            
            square = getSquare(pixels, boxW, i, j)
#             print(sqrOk(square, avg, midContrast))
            #first examine the square to see if it's worth calculating
            if sqrOk(square, avg, midContrast):
                
                theta, diff = checkPoint(square)
#                 print(diff)
                if absBiggerThan(diff, midContrast):

                    p,t,d = getBestInRegion(pixels, width, height, boxW, i, j, int(skipSize/2), skipSize, 'n', 'n', avg, lowContrast)
#                     print(i,j,p,d, highContrast)
                    if absBiggerThan(d, highContrast):
                        
                        outline = getOutline(pixels, width, height, boxW, maxLength, p[0], p[1], avg, lowContrast)
                        print("length: ",len(outline))
                        if len(outline) < 10:
                            continue
                        # split the outline here, before you fill it in.
                        # Pass it as a list.
                        
                        splitOutlines = splitOutline( outline )

                        outlineList.append(splitOutlines)
                        if len(outlineList) > 0:
                            fillOutline(outline, im2, fillCol)
#                             fillOutline(outline, im, fillCol)
#                             drawOutline(outline, im)
#                             im3 = im.copy()
#                             drawOutline(outline, im3)
#                             im3.show()

            print( "{0:.3f}% checked".format( (float((j * (width-boxW) + i) * 100)/((height-boxW) * (width-boxW)))  ) )
            
        print( "{0:.3f}% checked".format( (float((j - int(boxW/2)) * 100)/(height-boxW))  ) )

    print("100% checked")
    print("Printing points")
#     im2.show()
#     im.show()
#     im.save("test2.bmp")
#     return
    print(outlineList)
    for i1 in range(0, len(outlineList)):
        for i2 in range(0, len(outlineList[i1])):
            if len(outlineList[i1][i2]) > 4:
                drawOutline(outlineList[i1][i2], im)
                
#     im.show()
    
    start = 0
#     end = 1
    end = len(outlineList)
    ellipseList = []
    fiberList = []
    print("len(outlineList) = ", len(outlineList))
    
    for i2 in range(start, end):
        splitList = outlineList[i2]
        possEll = []
        for i3 in range(0, len(splitList)):
            list1 = splitList[i3]
#             print(i3, len(list1))
            a, b, h, k, t = solve(list1, minWidth/2)
            a = abs(a)
            b = abs(b)
#             print(i3, len(list1), a )
            if a > width/2 or h < 0 or k < 0 or h > width or k > height or a==0:
                continue
            h -= offset[0]
            k -= offset[1]
            fillEllipse(out, h, k, t, a, b, (255,255,255))
            print("save",len(ellipseList), a, b, h, k, t)
            ellipseList.append((h, k, t, a, b))

    return im, out, ellipseList


def getLightnessDiffference2(sqr, theta):
    theta += 0.001
    side1 = [0,0,0]
    side2 = [0,0,0]
    boxW = len(sqr[0])
    lBound = int(0 - boxW/2)
    hBound = boxW + lBound #ensures odd boxW's don't result in too short a range due to integer division
    s1Count = 0
    s2Count = 0
#     for i1 in range(lBound, hBound):
#         string = ""
#         for j1 in range(lBound, hBound):
#             i = i1 #+ 0.5
#             j = j1 #+ 0.5
#             if i>=0:
#                 string+=" "
#             string += str(i)+","
#             if j>=0:
#                 string += " "
#             string+= str(j)+"  "
#         print(string)
#         print("")
    for i1 in range(lBound, hBound):
        for j1 in range(lBound, hBound):
            if j1 == 0:
                continue
            i = i1 #+ 0.5
            j = j1 #+ 0.5
#                 print(i, j, k)
            
            tempTheta = getTheta( (i, j) )
            lThetaBnd = theta
            hThetaBnd = theta + math.pi
#             print(tempTheta, lThetaBnd, hThetaBnd, lThetaBnd - 2* math.pi, hThetaBnd - 2* math.pi )
            while (tempTheta < lThetaBnd) & (hThetaBnd > 2* math.pi):
                hThetaBnd -= 2* math.pi
                lThetaBnd -= 2* math.pi
            while (tempTheta > hThetaBnd) & (lThetaBnd < 0):
                hThetaBnd += 2* math.pi
                lThetaBnd += 2* math.pi
                
            distFromLine = sqrt((i**2 +j**2))*abs(math.tan(tempTheta - hThetaBnd))
#             if j == 0:
#                 print("___", i, j, math.atan( i / (j+0.000001) ))
#                 print(tempTheta , hThetaBnd)
            # rPixel goes between 0.5 and sqrt(0.5) depending on the dividing angle
            # r(    0 + n*pi/2) = 0.5
            # r( pi/4 + n*pi/2) = sqrt(0.5)
            rPixel = 0.5 + 0.2071*sin(2*theta)
#             if abs(distFromLine) < rPixel:
#             print(tempTheta/math.pi*180, i1, j1, distFromLine/rPixel)
#             print(i,j,int(tempTheta/math.pi *180+0.5), distFromLine )

#             print(abs(distFromLine))
            if (tempTheta > lThetaBnd) & ( tempTheta < hThetaBnd):
#                     im.putpixel((i+x,j+y), (0,255,0))
                for k in range(0,3):
                    if abs(distFromLine) < rPixel:
                        # a little more than half the width of a single pixel 
                        # I'm pretending the pixel's a circle, and r=0.56 gives it an equal area
                        side1[k] += sqr[i1][j1][k] * (distFromLine/(rPixel*2) + 0.5)
                        s1Count += distFromLine/(rPixel*2) + 0.5
                        side2[k] += sqr[i1][j1][k] * (0.5 - distFromLine/(rPixel*2))
                        s2Count += 0.5 - distFromLine/(rPixel*2)
                    else:
                        side1[k] += sqr[i1][j1][k]
                        s1Count += 1
            else:
#                     im.putpixel((i+x,j+y), (255,0,0))
                for k in range(0,3):
                    if abs(distFromLine) < rPixel:
                        # a little more than half the width of a single pixel 
                        # I'm pretending the pixel's a circle, and r=0.56 gives it an equal area
                        side2[k] += sqr[i1][j1][k] * (distFromLine/(rPixel*2) + 0.5)
                        s2Count += distFromLine/(rPixel*2) + 0.5
                        side1[k] += sqr[i1][j1][k] * (0.5 - distFromLine/(rPixel*2))
                        s1Count += 0.5 - distFromLine/(rPixel*2)
                    else:
                        side2[k] += sqr[i1][j1][k]
                        s2Count += 1

    difference = [0,0,0]
    for i1 in range(0,3):
#         print(side1[i1],side2[i1], s1Count, s2Count)
        difference[i1] = side1[i1]/s1Count - side2[i1]/s2Count
#     print(int(theta/math.pi * 180+0.5), side1[i1], side2[i1], s1Count, s2Count)
#     im.save("temp.bmp")
    return difference

#
#

()


import datetime
d1 = datetime.datetime.now()
# file = "Images/smallerTest.jpg"
file = "Images/tinyTest.jpg"
im2, out, outlines = standAlone(file)
out.show()
im2.show()
# im2.save("output.bmp")
# out.save("output2.bmp")
d2 = datetime.datetime.now()
diff = d2-d1
print(diff)
# 
# print(1/0)

def testStuff():
    im = Image.open("Images/ellipse.jpg")
    # im.show()
    im = im.convert('RGB')
    pixels = im.load()
    w,h = im.size
    sqrSize = 5
    sqr = getSquare(pixels, sqrSize, 29, 149)
    list1 = []
    
#     for i in range(-2,3):
#         for j in range(-2,3):
#             print(i,j,getTheta((i,j))/math.pi*180)
    
#     best = getBestAngle(sqr)/math.pi*180
#     print(best)
#     print(int(getLightnessDiffference2(sqr, (best-90)/180*math.pi)[0] * 10000) / 10000)
#     return
    prevDiff = 0.0
    for t in range(0,360):
        theta = t/180 * math.pi
        diff = getLightnessDiffference2(sqr, theta)[0]
#         print(diff)
#         diff2 = getLightnessDiffference(sqr, theta)[0]
        if (len(list1) == 0):
            list1.append((t,diff))
        elif (diff != list1[len(list1)-1][1]):
            list1.append((t,diff, diff - prevDiff))
        else:
            list1.append((t,diff, diff - prevDiff, "DUP"))
        prevDiff = diff
    # print(len(list1))
    for i in range(len(list1)):
        print(list1[i])
