'''

Christopher Hill


Apparently someone else has thought up the rotational binary search before:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.308.1650&rep=rep1&type=pdf
It's a description of finding skew in an image, by finding long lines which have been skewed, 
    and doing a "interval-halving binary search" to find the angle at which the maximum
    number of lines are in the same direction as that angle.
    Everything but the binary search is different though, and it doesn't help anything I'm doing.
    
    Actually it mentions estimating a confidence value in its binary thing by measuring
    both the min and the max in that binary search region, and returning max/min.
    If max/min ~= 1, then it's not good. I might be able to use that.

'''

from PIL import Image, ImageDraw
from colorSpread import getStats
from EllipseMath import sqrDist
from numpy.linalg import *
from numpy import *
import numpy
from scipy import optimize
from audioop import avg
import EllipseMath
from scipy.constants.constants import foot
from math import cos, sin, pi, sqrt
from functools import total_ordering
import os
from html.parser import interesting_normal
import datetime
from scipy.cluster.vq import ClusterError
from defer import AlreadyCalledDeferred

    
def withinTol( color1, color2, tol ):
    r = abs(color1[0] - color2[0])
    g = abs(color1[1] - color2[1])
    b = abs(color1[2] - color2[2])
    #print "r:" + str(r) + " g:"+ str(g) + " b:"+ str(b)
    return (r < tol[0]) & (g < tol[1]) & (b < tol[2])

#takes input of a coordinate (in (x,y) format, relative to the origin) on the regular xy plane
#gives theta as measured from the x axis. (polar)
def getTheta(coord):
    theta = math.atan( float(coord[1]) / (coord[0]+0.00001) )
#     print(theta)
    if coord[0] < 0:
        theta += math.pi
    elif coord[1] < 0:
        theta += math.pi * 2
    return theta


#function that will calculate the amount that side2 of a given 9x9 square is lighter than side1, 
#based on an input rotation of the centerline. the centerline starts on the x-axis.
#side1 is the centered at theta + pi/2
def getLightnessDiffference( sqr, theta ):
    theta += 0.001
    side1 = [0,0,0]
    side2 = [0,0,0]
    boxW = len(sqr[0])
    lBound = int(0 - boxW/2)
    hBound = boxW + lBound #ensures odd boxW's don't result in too short a range due to integer division
    s1Count = 0
    s2Count = 0

    for i1 in range(lBound, hBound):
        for j1 in range(lBound, hBound):
            if j1 == 0:
                continue
            i = i1
            j = j1
            
            tempTheta = getTheta( (i, j) )
            lThetaBnd = theta
            hThetaBnd = theta + math.pi
            twoPi = math.pi + math.pi
            while (tempTheta < lThetaBnd) & (hThetaBnd > twoPi):
                hThetaBnd -= twoPi
                lThetaBnd -= twoPi
            while (tempTheta > hThetaBnd) & (lThetaBnd < 0):
                hThetaBnd += twoPi
                lThetaBnd += twoPi
                
            distFromLine = sqrt((i**2 +j**2))*abs(math.tan(tempTheta - hThetaBnd))
#             distFromLine = ((abs(i)+abs(j))/1.372)*abs(tan(tempTheta - hThetaBnd))
            
            #pretending the pixel is a circle whose radius goes from 0.5 to 1.0 as theta goes from 0 to pi/2
            rPixel = 0.5 + 0.2071*sin(2*theta)
            dPixel = rPixel+rPixel
            partialPixelArea1 = distFromLine/dPixel + 0.5
            partialPixelArea2 = 0.5 - distFromLine/dPixel
            if (tempTheta > lThetaBnd) & ( tempTheta < hThetaBnd):

                for k in range(0,3):
                    if abs(distFromLine) < rPixel:
                        
                        side1[k] += sqr[i1][j1][k] * partialPixelArea1
                        s1Count += partialPixelArea1
                        side2[k] += sqr[i1][j1][k] * partialPixelArea2
                        s2Count += partialPixelArea2
                    else:
                        side1[k] += sqr[i1][j1][k]
                        s1Count += 1
            else:
#                     im.putpixel((i+x,j+y), (255,0,0))
                for k in range(0,3):
                    if abs(distFromLine) < rPixel:
                        # a little more than half the width of a single pixel 
                        # I'm pretending the pixel's a circle, and r=0.56 gives the circle an equal area to a 1x1 square
                        side2[k] += sqr[i1][j1][k] * partialPixelArea1
                        s2Count += partialPixelArea1
                        side1[k] += sqr[i1][j1][k] * partialPixelArea2
                        s1Count += partialPixelArea2
                    else:
                        side2[k] += sqr[i1][j1][k]
                        s2Count += 1

    difference = [0,0,0]
    for i1 in range(0,3):
        difference[i1] = side1[i1]/s1Count - side2[i1]/s2Count
        
    return difference
  
def getSquare( pixels, boxW, x, y ):
    square = []
    lBound = int(0 - boxW/2)
    hBound = boxW + lBound
    for i in range(lBound, hBound):
        square.append([])
        for j in range(lBound, hBound):
            try:
                square[len(square)-1].append( pixels[i + x, j + y] )
            except Exception:
                print(x,i,y,j)
                raise
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
    
def getSqrAvg(sqr):
    total = [0] * 3
    for c in range(0, 3):
        for i in range(0, len(sqr)):
            for j in range(0, len(sqr)):
                total[c] += sqr[i][j][c]
        total[c] /= len(sqr)**2 # might be able to remove this if more speed is needed. Probably wouldn't help much.
    return total

def getVector(p1,p2):
    return (p2[0]-p1[0], p2[1]-p1[1])

def vecMag(v):
    return sqrt(v[0]**2 + v[1]**2)

def getCosAngle( v1, v2 ):
    denom = vecMag(v1)*vecMag(v2)
    cosAngle = dot(v2,v1)/denom
    return cosAngle

def getNetDeltaAngle(outline, index, a, b, includeBounds = 0):
    # idx is starting index
    # a is number of places to look behind
    # b is the number to look ahead
    netAngle = 0;
    for i in range( index - a + 1 - includeBounds, index + b + includeBounds):
        p1 = outline[(i-1)%len(outline)]
        p2 = outline[i%len(outline)]
        v1 = getVector(p1,p2)
        
        p3 = outline[i%len(outline)]
        p4 = outline[(i+1)%len(outline)]
        v2 = getVector(p3,p4)

        t = getTheta(v2) - getTheta(v1)
        while t > pi:
            t -= 2*pi
        while t < -pi:
            t += 2*pi
#         print("    ", p1,p2,p3,p4, t)
        netAngle += t
    return netAngle


# l = [(0,0),(1,1),(2,1),(3,1),(4,0),(3,-1),(1,-1)]
# l = [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0)]
# print(getNetDeltaAngle(l, 2, 2, 4))
# print(1/0)

def getNextPoint(pixels, w, h, boxW, skipSize, i, j, t, veryHighContrast):
    # t is the angle at the current point, i and j are the location of the current point
    
    # current function makes 25 calls to checkPoint each time it's run. Try to use less than 8.
    
    # Along a circular path centered at the current point, of radius skipSize, check the lightness difference
    # between two points along that curve. The two points start out a small rotational distance away from the 
    # input current angle of contrast t, one on each side, and they move farther and farther from that center
    # angle until they have checked I think ~80% of the circle. I don't want it to check backwards.
    
    bestT = t
    bestDiff = veryHighContrast
    bestN=0
        
    stop = False
    
    nxtP = (0,0)
    
    ##this will work so long as skipSize > 1.5, or boxW > 3
    
    dt = 1.5*arctan(1/skipSize) #dt is 1.5 times the angle that holds a circumferential distance of one pixel.
    for c in range(0, int(pi*skipSize)): #check a range of points such that  
        
        #n is 1, -1, 2, -2, 3, -3, etc
#         n = ((-1)**c ) * int((c+3)/2)
        n = c+1
#         print(t, n, dt, n*dt, (t+n*dt)/pi*180, skipSize*dt)
        xL = i + boxW/2*cos(t + n*dt)
        yL = j + boxW/2*sin(t + n*dt)
        
        xM = i + boxW/2*cos(t)
        yM = j + boxW/2*sin(t)
        
        xR = i + boxW/2*cos(t - n*dt)
        yR = j + boxW/2*sin(t - n*dt)
        
        
        lSqrAvg = getSqrAvg( getSquare(pixels, 4, xL, yL) )
        mSqrAvg = getSqrAvg( getSquare(pixels, 4, xM, yM) )
        rSqrAvg = getSqrAvg( getSquare(pixels, 4, xR, yR) )
        
#         outerDiff = diffVec(lSqrAvg, rSqrAvg) # left-right difference
        lmDiff = diffVec(lSqrAvg, mSqrAvg) # left-middle difference
        rmDiff = diffVec(rSqrAvg, mSqrAvg) # right-middle difference
        
#         outDiffGood = biggerThan(outerDiff, highContrast)
        lmDiffGood = biggerThan(lmDiff, bestDiff)
        rmDiffGood = biggerThan(rmDiff, bestDiff)
        bestN = 0
        if lmDiffGood:
            bestN += n

        if rmDiffGood:
            bestN -= n

        print(lmDiffGood, rmDiffGood, bestN, t + bestN*dt/2)

    bestT = t + bestN*dt
    nxtP = (int(i + skipSize*cos(bestT)), int(j + skipSize*sin(bestT)))
    print(i,j,t,bestT*180/pi, nxtP, ((-1)**int(pi*skipSize-1) ) * int((int(pi*skipSize-1)+3)/2))

    return nxtP, bestT
    
    
def getOutline(pixels, w, h, boxW, maxLength, i, j, avg, stdev):
    print("Tracing outline, starting at", i, j)
    veryHighContrast = [x/3 for x in stdev]
    lowContrast = [x/8 for x in stdev]
    '''
    input is the first boundary point.
    '''
    skipSize = 1.7 * sqrt(2)
    outline = []
    prevP = (i, j)
    firstP = prevP
    outline.append(prevP)

    footprints = numpy.zeros(shape=(w,h))
#####
    sqr = getSquare(pixels, boxW, i, j)
    t = getBestAngle(sqr)
#####
    cutOffSqrDist = (3*skipSize)**2
#     print(firstP,"__")

    while True:
        if len(outline) > 15:
            dist = sqrDist(prevP, outline[0])

            if dist <= cutOffSqrDist:
                print(firstP, "close to start", len(outline))
                break
            elif dist > maxLength**2:
                print(firstP, "too far away", len(outline))
                break
            if len(outline) > (maxLength / skipSize)*3:
                print(firstP, "too many points (", len(outline))
                break

####
#         t = getBestAngle(getSquare(pixels, boxW, i, j))
#          
#         nxtP, t = getNextPoint(pixels, w, h, boxW, skipSize, i, j, t, veryHighContrast)
####
        i += skipSize * math.cos(t)
        j += skipSize * math.sin(t)
 
        nxtP, t, diff = getBestInRegion(pixels, w, h, boxW, i, j, 2, skipSize, prevP, avg, lowContrast)
        nxtP = (int(nxtP[0]), int(nxtP[1]))
####

        i,j = nxtP

        
        if nxtP == (0,0):
            print(firstP, "No valid next point", prevP, len(outline))
#             outline = []
            break
        if nxtP == prevP:
            print(firstP, "Next point is same as current point", prevP, len(outline))
            outline = []
            break

        looping = False
        
        line = getLineMatrix( prevP, nxtP)
        for p in line:
            if footprints[p[0]][p[1]] != 0:
                looping = True
        if looping:
            # means that nxtP and prevP are on either side of a previously drawn line; means that that section
            # has been traced before, and so the loop needs to stop.
            # so backtrack until the examined point is far enough from any other points, and
            # then re-define the list to cut off everything afterwards - 1.
            start = 0
            for start in range(len(outline)-4, -1, -1):
#                     print(start, sqrDist(outline[start], outline[len(outline)-1]), cutOffSqrDist / 5)
                if sqrDist(outline[start], outline[len(outline)-1]) < cutOffSqrDist / 5:
                    break
            outline = outline[start:]
            
            print(firstP, "Went into loop", len(outline))
            break
            
        if ((i - boxW/2 < 0)
            |(j - boxW/2 < 0)
            |(i + boxW/2 > w)
            |(j + boxW/2 > h)):
            
            print(firstP, "Went off-screen", len(outline))
            break
        
        outline.append(nxtP)
        for p in line: # this marks footprints both at and between each point
            footprints[p[0]][p[1]] = len(outline)
        prevP = nxtP
        
        '''
        add angle keeper-tracker to prevent tracing a void
        '''
        
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
        diff = getLightnessDiffference(sqr, usedTheta)
        halfPrevMovement /= 2.0
        if biggerThan(diff, (0,0,0)):
            usedTheta -= halfPrevMovement
        else:
            usedTheta += halfPrevMovement
        i+=0

    return usedTheta + math.pi / 2

def checkPoint( square ):
    
    theta = getBestAngle(square)
    diff = getLightnessDiffference(square, theta)
    
    return theta, diff


def getBestInRegion( pixels, width, height, boxW, x, y, searchSize, skipSize, prevP, avg, tol ):
    bestDiff = [0,0,0]
    bestT = 0
    bestP = (0,0)
    r = searchSize # x pixels away from the starting pixels, inclusive
    count = 0
    skipSizeSqrd = skipSize**2
    for j in range(-r, r+1, 1 + int(r/3+0.5)):
        for i in range(-r, r+1, 1 + int(r/3+0.5)):
# #     if x + (boxW/2 + 1) >= width:
# #         break
# #     elif y + (boxW/2 + 1) >= height:
# #         break
# #     elif x - (boxW/2 + 1) < 0:
# #         break
# #     elif y - (boxW/2 + 1) < 0:
# #         break
# #     else:
#     try:
#         t = checkPoint(getSquare(pixels, boxW, x, y))[0]
#     except Exception:
#         ()
#     points = 6
#     semiRange = pi/4
#     for a in range( -int(points/2), int(points/2) ):
#             angle = semiRange * 2 / points * a
#     
# #     angle = t + pi/2
# #     for c in range(-3,3):
# #             i = int(cos(angle) * c)
# #             j = int(sin(angle) * c)
#             i = int(cos(angle) * r)
#             j = int(sin(angle) * r)
            
            if x + i + (boxW/2 + 1) >= width:
                break
            elif y + j + (boxW/2 + 1) >= height:
                break
            elif x + i - (boxW/2 + 1) < 0:
                i+= (boxW/2 + 1)
            elif y + j - (boxW/2 + 1) < 0:
                j+= (boxW/2 + 1)
            else:
                pointIsGood = False
                if prevP != 'n':
                    if sqrDist(prevP, (x + i, y + j)) > skipSizeSqrd:
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
                    if prevP == (81, 100):
                        print(theta, diff, bestDiff, i, j)
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
#     get points on the outline of the specified ellipse
#     create polygon using those points
#     draw/fill polygon
#     '''
    pnts = getEllipseOutline(x0, y0, alpha, a, b, True)
    draw = ImageDraw.Draw(im)
    draw.polygon(pnts, fill=col)

def roundToNearest(x):
    return int(round(x) - .5) + (x > 0)

def sumVec(v1,v2):
    return (v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2])

def diffVec(v1,v2):
    v = (v1[0]-v2[0], v1[1]-v2[1], v1[2]-v2[2])
    return v

def getVecInOutline(i1, i2, outline):
    return (outline[ i1 ][0]-outline[ i2 ][0], outline[ i1 ][1]-outline[ i2 ][1])

    
def isBreakPoint(outline, i):
    
    '''
    make it take into account the sine of the angle (cross product?), and from that deduce the quadrant that the angle
    is in, and thereby be able to distinguish 179 degrees from 1 degree.
    ?
    might not be necessary
    cross product is expensive
    '''
    i3a = (i-3) % len(outline)
    i2a = (i-2) % len(outline)
    i1a = (i-1) % len(outline)
    i0 = i
    i1b = (i+1) % len(outline)
    i2b = (i+2) % len(outline)
    i3b = (i+3) % len(outline)

    a1 = getCosAngle(getVecInOutline( i0, i1a, outline), getVecInOutline( i0, i1b, outline))
    a2 = getCosAngle(getVecInOutline( i0, i2a, outline), getVecInOutline( i0, i2b, outline))
    a3 = getCosAngle(getVecInOutline( i0, i3a, outline), getVecInOutline( i0, i3b, outline))
    
    angle = (a1 + a2 + a3)/3
    
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
    
def getFullOutline(outline):
    '''
    THIS IS NOT AN ORDERED LISTING!!!!
    '''
    fullOutline = []
    for i in range(0, len(outline)):
        x1 = outline[i][0]
        y1 = outline[i][1]
        x2 = outline[(i+1) % len(outline)][0] # modulo so that it wraps to the beginning when it considers the last point
        y2 = outline[(i+1) % len(outline)][1] # in the outline
        
        if (x2-x1) != 0:
            # x is dependent
            myx = (y2-y1)/(x2-x1)
            by = y1 - myx*x1
            fullOutline.append((x1,y1))
            for x0 in range(min(x1*10, x2*10), max(x1*10, x2*10)):
                x = x0/10
    #                 print(x, m*x+b)
                if (int(x), int(myx*x+by)) != fullOutline[len(fullOutline)-1]:
                    fullOutline.append((int(x), int(myx*x+by)))
        if (y2-y1) != 0:
            # y is dependent
            mxy = (x2-x1)/(y2-y1)
            bx = x1 - mxy*y1
            fullOutline.append((x1,y1))
            for y0 in range(min(y1*10, y2*10), max(y1*10, y2*10)):
                y = y0/10
    #                 print(x, m*x+b)
                if (int(mxy*y+bx), int(y)) != fullOutline[len(fullOutline)-1]:
                    fullOutline.append((int(mxy*y+bx), int(y)))

    return fullOutline

def getNearbyAvg(matrix, p, r):
    if r != int(r):
        raise Exception("input r must be an integer")
    x0, y0 = p[:]
    w = 2*r + 1
#     sqr = np.array([[0]*w]*w)
    sum1 = 0
    for x in range(0, w):
        for y in range(0, w):
#             sqr[x][y] = matrix[x0-r+x][y0-r+y]
            sum1 += matrix[x0-r+x][y0-r+y]
    sum1 /= w**2
    return sum1

# cur = (0,0)
# nxt = (5,2)
# nxtnxt = (10, 6)
# v1 = getVector(cur, nxt)
# v2 = getVector(nxt, nxtnxt)
# # print(getTheta(v1))
# print(getTheta(v1), getTheta(v2), getTheta(v2) - getTheta(v1))#, cos(getTheta(v2) - getTheta(v1)), dot(v1,v2)/(1*sqrt(2)))
# print(1/0)

def getClusters(data, eps):
    clusters = []
    for p1 in data:
        skip = False
        for c in clusters:
            if c.__contains__(p1):
                skip = True
                break
        if skip:
            continue
        c = []
        c.append(p1)
        for p2 in c:
            nearby = regionQuery(data, p2, eps)
            for p3 in nearby:
                if not c.__contains__(p3):
                    c.append(p3)
        clusters.append(c)
    return clusters
        
# 
# #clustering algorithm from 
# def DBSCAN(Data, eps):
#     Clusters = []
#     for i1 in range(0, len(Data)):
#         if Data[i1] == 0:
#             continue
#         P = Data[i1]
#         Data[i1] = 0
#         NeighborPts = regionQuery(Data, P, eps)
#         print(i1, P, NeighborPts)
#         Clusters.append([])
#         Clusters[len(Clusters)-1].append(P)
#         i2 = -1
#         while i2 < len(NeighborPts)-1:
#             i2+=1
#             p0 = NeighborPts[i2]
#             if p0 != 0:
#                 near = regionQuery(Data, p0, eps)
#                 if not near.__contains__(p0):
#                     NeighborPts.append(p0)
#             else:
#                 continue
#             
#             
#             alreadyAdded = False
#             for c in Clusters:
#                 for p1 in c:
#                     if P0 == p1:
#                         alreadyAdded = True
#                         break
#                 if alreadyAdded:
#                     break
#                 
#             if not alreadyAdded:
#                 Clusters[len(Clusters)-1].append(P0)
#     return Clusters
 
def regionQuery(Data, P, eps):
    nearby = []
    epsSqr = eps**2
    for p0 in Data:
        if (p0 != 0) and (P != p0) and (sqrDist(p0, P) < epsSqr):
            nearby.append(p0)
    return nearby


# im = Image.new("RGB", (200,200), (0,0,0))
# l = []
# l.append((25,25))
# l.append((27,27))
# l.append((35,23))
# l.append((42,30))
# 
# l.append((50,100))
# l.append((60,100))
# l.append((70,100))
# l.append((80,100))
# 
# l.append((125,190))
# 
# l.append((185,50))
# l.append((175,55))
# l.append((185,55))
# l.append((185,65))
# l.append((175,50))
# l.append((185,45))
# for p in l:
#     im.putpixel(p, (255,255,255))
# im.show()
# # clusters = DBSCAN(l, 20)
# clusters = tryAgain(l, 20)
# print(len(clusters))
# for i in range(0, len(clusters)):
#     for p in clusters[i]:
#         col = int(i/len(clusters)*100)
#         im.putpixel(p, (col+155*((i+2)%3),col+155*((i+1)%3),col+155*(i%3))) #(255-col)*(i%3)
# im.show()
# print(1/0)

def splitOutline(im, outline, outputIm1 = 0,  outputIm2 = 0 ):
    print("Entered splitOutline")
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
    
#     # initial idea, works kinda but not well enough
#     '''
#     if a point in a list has too high of a curvature, replace that list (in splitlist) with the preceding part of the list,
#     and add a list containing the remaining points to the end. If either section is too small, don't add it.
#     '''
#     start = -1
#     for i1 in range(0, len(outline)):
#         if isBreakPoint(outline, i1):
#             start = i1
#             break
#     if start == -1:
#         splitList.append(outline)
#         return splitList
#     
#     bP = [] #break points
#     print("Start:",start)
#     for i2 in range(0, len(outline)):
#         if isBreakPoint(outline, (i2 + start)%len(outline)):
#             bP.append((i2 + start)%len(outline))
#             
#     print("BP: ", bP)
#     
#     for i3 in range(0, len(bP)):
#         if abs(bP[i3]+1 - bP[(i3-1)%len(bP)]) > 0:
# #             splitList.append(outline[bP[i-1]:bP[i]])
#             splitList.append(getSlice(outline, bP[(i3-1)%len(bP)], bP[i3]+1))
#     print("SplitList:", splitList)
# #     im = Image.new('RGB',(500,600), (0,0,0))
# #     
# #     for i in range(0, len(splitList)):
# #         for j in range(0, len(splitList[i])):
# #             p = (splitList[i][j][0],splitList[i][j][1])
# #             c = int(i/len(splitList) * 255)
# #             c = i%2 * 255
# #             col = (c,0,255-c)
# #             im.putpixel(p, col)
# #     im.show()

    # new attempt, using watershed
    
    im0 = im.copy()
    im1 = im.copy()
    pixels0 = im0.load()
    w,h = im.size
    
    if outputIm1 == 0:
        outputIm1 = Image.new("RGB", (w, h), (0,0,0))
    if outputIm2 == 0:
        outputIm2 = Image.new("RGB", (w, h), (0,0,0))
    
    out1Pixels = outputIm1.load()
        
    tracedOutline = getFullOutline(outline)
    
    for i1 in range(0, len(tracedOutline)):
        im0.putpixel(tracedOutline[i1], (255,0,0))
    for i1 in range(0, len(outline)):
        im1.putpixel(outline[i1], (255,0,0))
    
    interiorPoint = 'bad point'
    
    pointsIncludingCenter = []
    xMin = w
    xMax = 0
    yMin = h
    yMax = 0
    for i1 in range(0, len(outline)):
        if outline[i1][0] < xMin:
            xMin = outline[i1][0]
        elif outline[i1][0] > xMax:
            xMax = outline[i1][0]
        if outline[i1][1] < yMin:
            yMin = outline[i1][1]
        elif outline[i1][1] > yMax:
            yMax = outline[i1][1]
    
    lBnd = xMin - int((xMax-xMin)/2)
    rBnd = xMax + int((xMax-xMin)/2)
    upBnd = yMax + int((yMax-yMin)/2) # up in the coordinate system, down on the screen
    dwBnd = yMin - int((yMax-yMin)/2)
    print("Bounds: ", rBnd, lBnd, upBnd, dwBnd)
    if lBnd < 0:
        shift = -lBnd
        lBnd = 0
        rBnd += shift
    elif rBnd > w-1:
        shift = rBnd - (w-1)
        rBnd = w-1
        lBnd -= shift
    if dwBnd < 0:
        shift = -dwBnd
        dwBnd = 0
        upBnd += shift
    elif upBnd > h-1:
        shift = upBnd - (h-1)
        upBnd = h - 1
        dwBnd -= shift
    
    if ((lBnd < 0) or (rBnd >= w) or (dwBnd < 0) or (upBnd >= h)):
        print("Found an outline running entire length of image. Can't split it.")
        return []
    
#     pointsIncludingCenter.append((lBnd + 2, dwBnd+2))
    pointsIncludingCenter.append((int((xMax+xMin)/2), int((yMax+yMin)/2)))
    i = 0
    try:
        print("initial point: ", pointsIncludingCenter[i])
        while i < len(pointsIncludingCenter):
            outputIm1.putpixel(pointsIncludingCenter[i], (0,255,255))
            right = (pointsIncludingCenter[i][0] + 1, pointsIncludingCenter[i][1])
            left = (pointsIncludingCenter[i][0] - 1, pointsIncludingCenter[i][1])
            up = (pointsIncludingCenter[i][0], pointsIncludingCenter[i][1] + 1)
            down = (pointsIncludingCenter[i][0], pointsIncludingCenter[i][1] - 1)
#             print(pointsIncludingCenter[i], "|", right, rBnd, (right[0] < rBnd),"|", left, lBnd, (left[0] > lBnd),"|", up, upBnd, (up[0] < upBnd),"|", down, dwBnd, (down[0] > dwBnd))
            
            if (right[0] < rBnd) and (pixels0[right] != (255,0,0)):
                pointsIncludingCenter.append(right)
                im0.putpixel(right, (255,0,0))
                
            if (left[0] > lBnd) and (pixels0[left] != (255,0,0)):
                pointsIncludingCenter.append(left)
                im0.putpixel(left, (255,0,0))
                
            if (up[1] < upBnd) and (pixels0[up] != (255,0,0)):
                pointsIncludingCenter.append(up)
                im0.putpixel(up, (255,0,0))
                
            if (down[1] > dwBnd) and (pixels0[down] != (255,0,0)):
                pointsIncludingCenter.append(down)
                im0.putpixel(down, (255,0,0))
            i += 1
    except Exception:
        im0.show()
        raise
    
    shape = []
    
    areaSquareEnclosingOutline = (xMax-xMin)*2 * (yMax-yMin)*2
#     if outline[0] == (77, 25):
#         print(areaSquareEnclosingOutline, i)
#         im0.show()
#         im1.show()
#     print("area = ", i, "max = ", areaSquareEnclosingOutline, areaSquareEnclosingOutline/4)
    if i < areaSquareEnclosingOutline/4:
        shape = pointsIncludingCenter
        interiorPoint = (int((xMax+xMin)/2), int((yMax+yMin)/2))
    else:
        for i1 in range(0, len(outline)):
            # for a sequence of three points which are convex, which I can use to get an internal point.
            cur = outline[i1]
            nxt = outline[(i1 + 1) % len(outline)]
            nxtnxt = outline[(i1 + 2) % len(outline)]
#             v1 = getVector(cur, nxt)
#             v2 = getVector(nxt, nxtnxt)
            avg = ( (cur[0] + nxt[0] + nxtnxt[0])/3, (cur[1] + nxt[1] + nxtnxt[1])/3 )
            if pixels0[avg] != (255,0,0):
                interiorPoint = ( int((cur[0]+nxtnxt[0])/2), int((cur[1]+nxtnxt[1])/2) )
#                 im1.putpixel(cur, (0,0,255))
#                 im1.putpixel(nxt, (0,128,255))
#                 im1.putpixel(nxtnxt, (0,255,255))
                print("insidePoint:",cur,nxt,nxtnxt,interiorPoint)
                shape.append(interiorPoint)
                i = 0
                
                try:
                    while i < len(shape):
                        outputIm1.putpixel(shape[i], (0,255,255))
                        right = (shape[i][0] + 1, shape[i][1])
                        if pixels0[right] != (255,0,0):
                            shape.append(right)
                            im0.putpixel(right, (255,0,0))
                            
                        left = (shape[i][0] - 1, shape[i][1])
                        if pixels0[left] != (255,0,0):
                            shape.append(left)
                            im0.putpixel(left, (255,0,0))
                            
                        up = (shape[i][0], shape[i][1] - 1)
                        if pixels0[up] != (255,0,0):
                            shape.append(up)
                            im0.putpixel(up, (255,0,0))
                            
                        down = (shape[i][0], shape[i][1] + 1)
                        if pixels0[down] != (255,0,0):
                            shape.append(down)
                            im0.putpixel(down, (255,0,0))
                            
                        i += 1
                        
                except Exception:
#                     shape = []
#                     im0 = im.copy()
#                     im1 = im.copy()
#                     pixels0 = im0.load()
#                     for i1 in range(0, len(tracedOutline)):
#                         im0.putpixel(tracedOutline[i1], (255,0,0))
#                     for i1 in range(0, len(outline)):
#                         im1.putpixel(outline[i1], (255,0,0))
#                     continue
                    return splitList
                    outputIm1.show()
                    im0.show()
#                     im0.save("crap.bmp")
                    im1.show()
                    print("\n\nIn shape flood-fill exception", i, shape[i])
                    splitList.append(outline)
                    return 1/0
#                     raise
                    return splitList
                break
                
#                 if nxt[0] == cur[0] or nxtnxt[0] == nxt[0]:
#                     continue
#                 slope1 = (nxt[1]-cur[1])/(nxt[0]-cur[0])
#                 slope2 = (nxtnxt[1]-nxt[1])/(nxtnxt[0]-nxt[0])
#                 if (slope1*slope2 > 0) and (slope1 > slope2):
#                     if sqrDist(cur, nxtnxt) < 8:
#                         continue
#                     
    
    if interiorPoint == 'bad point':
        return splitList
#     im0.show()
#     im0.save("borderedThingy.bmp")
    
    
    #populate distance matrix
    distMatrix = array([[0]*h]*w)
    for p in shape:
        minSqrdDist = w*w
        for o in tracedOutline:
            dist = sqrDist(p, o)
            if dist < minSqrdDist:
                minSqrdDist = dist
        distMatrix[p[0]][p[1]] = minSqrdDist
    
    if False != True:
#         im2 = Image.new("RGB", (w,h), (0,0,0))
        maxDist = 0
        maxDist = 0
        minDist = w*w
        for x in range(0, w):
            for y in range(0, h):
                if distMatrix[x][y] > maxDist:
                    maxDist = distMatrix[x][y]
                elif distMatrix[x][y] < minDist:
                    minDist = distMatrix[x][y]
        
        scale = 255/(maxDist - minDist)

        for x in range(0, w):
            for y in range(0, h):
                c = int(scale * distMatrix[x][y])
                c1 = out1Pixels[(x,y)][0]
                if c > c1:
                    outputIm1.putpixel((x,y), (c,c,c))
                
#         outputIm2.show()
#         outputIm2.save("pretty.bmp")
    
    # find minima
    allMinima = []
    sqrW = 12
    
    print("Finding minima")
    for i in range(0, len(shape), 10*sqrW):
        bestP = shape[i]
        
        for cutoff in range(0, int(sqrt(len(shape)))):
            currentP = bestP
#             minimaPathTrace.putpixel(currentP, (255,0,0))
            
            best = getNearbyAvg(distMatrix, currentP, int(sqrW/2))
            for i1 in range(-1,2):
                for i2 in range(-1,2):
                    if i1 == 0 and i2 == 0:
                        continue
                    avg = getNearbyAvg(distMatrix, (bestP[0] + i1*sqrW, bestP[1] + i2*sqrW), int(sqrW/2))
                    if avg > best:
                        bestP = (bestP[0] + i1*sqrW, bestP[1] + i2*sqrW)
                        best = avg
            if bestP == currentP:
                break
        
#         dontAdd = False
#         for m in minima:
#             if sqrDist(bestP, m) < 2*sqrW:
#                 dontAdd = True
#                 break
#         if not dontAdd:
        allMinima.append(bestP)
        
#             minimaPathTrace.putpixel(currentP, (0,0,255))
#     minimaPathTrace.show()
#     outputIm1.show()
    
    # now clustering algorithm, to average together all points which are within some small distance of each other.
    # Groups a, b, and c if aRb and bRc, even if not aRc.
    clusters = getClusters(allMinima, 20)
    minima = []
    for c in clusters:
        totX = 0
        totY = 0
        for p in c:
            totX += p[0]
            totY += p[1]
        minima.append((int(totX/len(c)),int(totY/len(c))))
#     minima = allMinima
    for m in allMinima:
        try:
            outputIm1.putpixel(m, (255,255,0))
        except Exception:
            ()
    for m in minima:
        try:
            outputIm1.putpixel(m, (255,0,0))
        except Exception:
            ()
        
    
    breakPoints = []
    print("minima", minima)
    if len(minima) > 2:
        for i4 in range(0, len(tracedOutline)):
            p = tracedOutline[i4]
            # find the closest two minima
            min1 = (0,0)
            min1Dist = w*w
            min2 = (0,0)
            min2Dist = w*w
            for m in minima:
                dist = sqrDist(p, m)
                if dist < min2Dist:
                    if dist < min1Dist:
                        min2Dist = min1Dist
                        min2 = min1
                        min1Dist = dist
                        min1 = m
                    else:
                        min2Dist = dist
                        min2 = m
    #         print(p, abs(sqrDist(p, min1) - sqrDist(p, min2)))
            if abs(sqrDist(p, min1) - sqrDist(p, min2)) < 60:
#                 print(p, "###############################")
                # p is a breakpoint, find nearest points in outline
                for i in range(0, len(outline)):
                    # if p is between (in both x and y directions) the current point and the next one
                    prv = outline[i]
                    nxt = outline[(i+1)%len(outline)]
                    if (    p[0] >= min(prv[0], nxt[0]) 
                        and p[0] <= max(prv[0], nxt[0])
                        and p[1] >= min(prv[1], nxt[1]) 
                        and p[1] <= max(prv[1], nxt[1])):
                        isContained = False
                        for i1 in range(0, len(breakPoints)):
                            if abs(breakPoints[i1] - i) < 10:
                                isContained = True
                                break
                        if not isContained:
                            breakPoints.append(i)
#     print("___________________________________",breakPoints)
                
    if len(breakPoints) == 0:
        splitList.append(outline)
        return splitList
    
    for i1 in range(0, len(breakPoints)):
        if abs(breakPoints[i1]+1 - breakPoints[(i1-1)%len(breakPoints)]) > 0:
#             splitList.append(outline[bP[i-1]:bP[i]])
            splitList.append(getSlice(outline, breakPoints[(i1-1)%len(breakPoints)], breakPoints[i1]+1))
#     if outputIm2 != 0:
#         for i2 in range(0, len(splitList)):
#             drawPerimeter(splitList[i2], outputIm2, (0,255,0))
    #     outputIm2.show()
    return splitList

def ellipsesAreSame(e1, e2):
    # returns true if the ellipses are close/similar, in some metric
    #not yet implemented
    return False

def avgEll( e1, e2 ):
    return 0
# fix this, what if t1 = 1 and t2 = 179?
# or if a and b are swapped?
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


def drawPerimeter( outline, image, col ):
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
        p1 = optimize.leastsq(errfunc, p0, args=(xVec,yVec))[0]
#         p1, success = optimize.leastsq(errfunc, p0, args=(xVec,yVec))
        # p1, success = scipy.optimize.root(errfunc, p0, args=(xVec,yVec), method = 'lm')
        a,b,h,k,t = p1[:]
        if b > a:
            t += pi/2
            temp = a
            a = b
            b = temp
        return a,b,h,k,t
    except Exception:
        print("\nDied", Exception.args)
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

def saveData(listOfLists, fileName):
    with open(fileName, 'w') as file:
        # go through the list
        for i in range(0, len(listOfLists)):
            # go through each set of similar split outlines
            for j in range(0, len(listOfLists[i])):
                # go through each outline in that set
                for k in range(0, len(listOfLists[i][j])):
                    # go through each point in that outline
                    p = listOfLists[i][j][k]
                    file.write(str(p[0]) + " " + str(p[1])+ "\n")
                file.write('...\n')
            file.write('___\n')
    file.close()
    
def loadData(fileName):
    listOfLists = []
    # go until you hit EOF ?
    with open(fileName, 'r') as file:
        # fill the list
        readSplitOutlines = True
        while readSplitOutlines:
            # fill a set of similar split outlines
            splitOutlines = []
            readOutline = True
            while readOutline:
                # fill each outline in that set
                outline = []
                while True:
                    # read each point
                    words = file.readline().split()
                    if len(words) == 0:
                        readOutline = False
                        readSplitOutlines = False
                        break
                    if words[0] == "...":
                        break
                    if words[0] == "___":
                        readOutline = False
                        break
                    p = (int(words[0]),int(words[1]))
#                         print(p)
                    outline.append(p)
                if outline != []:
                    splitOutlines.append(outline)
            
            if splitOutlines != []:
                listOfLists.append(splitOutlines)
        file.close()
    return listOfLists

# l = []
# l1 = [[(0,1),(1,0),(23,0)]]
# l2 = [[(2,3),(3,2)]]
# l3 = [[(4,5),(5,4)]]
# l.append(l1)
# l.append(l2)
# l.append(l3)
# print(l)
# saveData(l, "mrFile.txt")
# l = loadData("mrFile.txt")
# print(l)
# print(1/0)

def standAlone( imPath, minWidth ):
    d1 = datetime.datetime.now()
    
    # took ~2:52:10 for a 2000x4000 image, when using the 25 box checker, skipping by 2. (so 6 points? or 12?)

#     imPath = "Images/rods.jpg"
#     imPath = "Images/testPict.jpg"
#     imPath = "Images/testPictTiny.jpg"
#     imPath = "Images/testPicFlare.jpg"
#     imPath = "Images/testPict2.jpg"
#     imPath = "Images/test2.jpg"
#     imPath = "SplitTest.bm"
#     imPath = "Images/ellipse.jpg"
#     imPath = "Images/ellipseBlurred.jpg"
#     imPath = "Images/FiberImages/BOSCH/BOSCH_EGPNylon66_50wt_LGF_2mm_F41_90_W_90_L_3_R(color).jpg"
#     imPath = "Images/FiberImages/44.5_LCF/LCF_EGP_44.5wt__2sec_79Deg_xz-plane_C6_0_W_40_L_50x_~1.5mm_Fixed_R(color).jpg"
#     imPath = "TestCases/full/ltBlue1.jpg"
#     imPath = "TestCases/noise/pittedLtBlue1.jpg"
#     imPath = "TestCases/full/blue1.jpg"
    imName = os.path.split(imPath)[1]
    im = Image.open(imPath)
#     im = Image.new("RGB", (100, 100), (40,80,160))
#     im.show()
#     return
    im2 = im.copy()
    
    width, height = im.size
    out = Image.new("RGB", (width, height), (0,0,0))
    pixels = im.load()
    
    print("Image loaded.")
    
    boxW = int(minWidth/3) # width of the square used to find boundaries # worked(ish) at boxW = 40/3 = 13
    boxW = 13
    if boxW % 2 == 0:
        boxW += 1
#     maxLength = 40*minWidth
    maxLength = width if (width > height) else height

    
    print("Getting statistics...")
    stdev, avg = getStats(pixels, width, height)
    
    #the tolerances for picking a "good" point
    highContrast = [ 2*x/8 for x in stdev ]
    midContrast = [ 5*x/32 for x in stdev ]
    lowContrast = [ x/8 for x in stdev ]
    
    fillCol = avg
    for i in range(0, len(fillCol)):
        fillCol[i] = int(fillCol[i] - 4*stdev[i]/7)
    fillCol = tuple(fillCol)
    
    print("avg = ", avg)
    print("stdev = ", stdev)
    
    #LATER - increase the iterator maybe, once the ellipse finding works.
    # it probably could be much higher than it is now. This loop only needs to find 1 dot
    # on every ellipse; not all of the dots.
    # 14 missed an ellipse
    # it'l have to be dependent on the scale of the ellipses in the image
    skipSize = int(boxW / 2)

    outlineList = []

    # center image on dark background
    border = 2*boxW

    background = Image.new('RGB', (width + 2*border, height + 2*border), fillCol)
    bg_w, bg_h = background.size
    offset = (int((bg_w - width) / 2), int((bg_h - height) / 2))
    background.paste(im2, offset)
    background.save("bordered"+imName[:len(imName)-3]+"bmp")
#     background.show()
#     return
    # re-load these variables
    im2 = background
    im = im2.copy()
    original = im2.copy()
    width,height = im2.size
    pixels = im2.load()
    
    outputIm1 = Image.new("RGB", (width, height), (0,0,0))
#     outputIm2 = Image.new("RGB", (width, height), (0,0,0))
    
    
#     im.show()
#     return
###############

#     i,j = 200, 62
#     i,j = 211, 53
#     i,j = 36, 18
#  
#     t = getBestAngle(getSquare(pixels, boxW, i, j))
#     p, t = getNextPoint(pixels, width, height, boxW, 1.7 * sqrt(2), i, j, t, [ x/3 for x in stdev ])
#     print(i,j, p, t )
#     im.putpixel((i,j), (255,255,0))
#     im.putpixel(p, (0,255,0))
#     im.show()
#     return
 
#     outline = getOutline(pixels, width, height, boxW, maxLength, i,j, avg, stdev)
#     print(len(outline))
#     
#     drawOutline(outline, im)
#     im.putpixel((i,j), (255,255,0))
#     im.show()
#     return
     
    startX = offset[0]
    startY = offset[1]
#     startX = 74
#     startY = 26
     
    for j in range(startY, height - offset[1], skipSize):
 
        for i in range(startX, width - offset[0], skipSize):
            '''
            1. check point
            2. if point is good, try to trace outline
            3. if there are enough points in the outline, add it to the list
            '''
            #test on 401x302
            #took about 5.5 mins, most of which felt like the continuous loops
             
            square = getSquare(pixels, boxW, i, j)
#             print('\nBegininng of standAlone loop',i,j)
            #first examine the square to see if it's worth calculating
            if sqrOk(square, avg, midContrast):
                 
                #the first result, theta, is ignored here
                diff = checkPoint(square)[1]
 
                if absBiggerThan(diff, midContrast):
#                     print('Right before getBestInRegion',i,j)
                    p,t,d = getBestInRegion(pixels, width, height, boxW, i, j, int(skipSize/2), skipSize, 'n', avg, lowContrast)
#                     print('\tRight after getBestInRegion',p)
 
                    if absBiggerThan(d, highContrast):
#                         print('Right before getOutline',p)
                        outline = getOutline(pixels, width, height, boxW, maxLength, p[0], p[1], avg, stdev)
#                         print('t',i,j)
#                         drawOutline(outline, im)
#                         im.show()
#                         return
                        if len(outline) < 7 or getNetDeltaAngle(outline, 0, 0, len(outline)) > 0:
                            continue
                        
#                         if outline[0] != (77, 25):
#                             continue
                        
                        # split the outline here, before you fill it in.
#                         splitOutlines = splitOutline( im, outline, outputIm1, outputIm2 )
                        splitOutlines = splitOutline( im, outline, outputIm1 )
                        
                        print("First point in splitOutline: ", outline[0])
                        print("len: ", len(splitOutlines))
#                         fillOutline(outline, im, fillCol)
#                         drawOutline(outline, im)
#                         im.show()
#                         im3 = im.copy()
#                         drawOutline(outline, im3)
#                         im3.show()
#                         return
                        
                        if len(outlineList) > 0:
                            fillOutline(outline, im2, fillCol)
                        if splitOutlines == []:
                            continue
#                         splitOutlines = splitOutline( im, outline )
                        outlineList.append(splitOutlines)
                        print("SplitOutline Appended.\n")
                        
                        fillOutline(outline, im, fillCol)
                        drawOutline(outline, im)
#                         im3 = im.copy()
#                         drawOutline(outline, im3)
#                         im3.show()
#                         im.show()
#                         return
 
#             print( "\r{0:.3f}% checked".format( (float((j * (width-boxW) + i) * 100)/((height-boxW) * (width-boxW)))  ) )
             
#         print( "\r{0:.3f}% checked".format( (float((j - int(boxW/2)) * 100)/(height-boxW))  ) )
 
    print("100% checked")
    print("Printing points")
    outputIm1.show()
#     outputIm1.save("largerOutput.bmp")
#     outputIm2.show()
#     outputIm2.save("largerOutput.bmp")
     
#     saveData(outlineList,"file.txt")
    
#     im.show()
    print("Time to trace and split: ", datetime.datetime.now() - d1)
    
##################

#     outlineList = loadData("file.txt")
    

#     for i1 in range(0, len(outlineList)):
#         for i2 in range(0, len(outlineList[i1])):
#             if len(outlineList[i1][i2]) > 4:
#                 drawOutline(outlineList[i1][i2], im)
#                 length = len(outlineList[i1][i2])
#                 dist = 1
#                 for i3 in range(0, length, 1):
#                     t = getNetDeltaAngle(outlineList[i1][i2], i3, dist, dist, 1)
#                     if abs(t) > 0.5*pi:
#                         print(t, outlineList[i1][i2][i3], "    ****")
#                         im.putpixel(outlineList[i1][i2][(i3+1)%length], (0,0,255))
                        
#     for i in range(24,len(outlineList)):

#     indx = 27
#     indx = 0
#     drawOutline(outlineList[indx][0], im)
#     for i in range(0, len(outlineList[indx][0])):
#         t = getNetDeltaAngle(outlineList[indx][0], i, 2, 2, 1)
#         if abs(t) > 0.66*pi:
#             print(t, outlineList[indx][0][i], "    ****")
# #             im.putpixel(outlineList[indx][0][i], (0,0,255))
#         else:
#             print(t, outlineList[indx][0][i])

#     im.show()
#     im.save("outlined"+imName[:len(imName)-3]+"bmp")
    
#     return
    start = 0
    end = len(outlineList)
    ellipseList = []

    print("len(outlineList) = ", len(outlineList),"\n")
    
    for i2 in range(start, end):
        splitList = outlineList[i2]
#         possEll = []
#         print("Checking item", i2, "out of", len(outlineList)) #not really checking that number; remember that it's
                                                                #checking that collection of lists
        for i3 in range(0, len(splitList)):
            list1 = splitList[i3]
#             print(i3, len(list1))
            a, b, h, k, t = solve(list1, minWidth/2)
            a = abs(a)
            b = abs(b)
#             print(a,b,h,k,t)
#             print((a, width), (2*b, minWidth), (width, height))
#             imTemp = original.copy()
#             imTemp2 = original.copy()
#             drawOutline(list1, imTemp)
#             drawOutline(list1, imTemp2)
#             fillEllipse(imTemp2, h, k, t, a, b, (255,255,255))
#             imTemp.show()
#             imTemp2.show()
#             input("")
            
#             print(i3, len(list1), a )
            if (a > width) or (2*b < minWidth) or (h <= 0) or (k < 0) or (h > width) or (k > height):
                continue
            h -= offset[0]
            k -= offset[1]
            fillEllipse(out, h, k, t, a, b, (255,255,255))
            adjustedList = [ (p[0]-offset[0], p[1]-offset[1]) for p in list1 ]
            try:
                drawOutline(adjustedList, out)
            except Exception:
                ()
            print("saving ellipse #", len(ellipseList), a, b, h, k, t)
            ellipseList.append((h, k, t, a, b))

    print("Time to find best fits: ", datetime.datetime.now() - d1)
    return im, out, ellipseList



def main():
    d1 = datetime.datetime.now()
    # file = "Images/smallerTest.jpg"
#     file = "Images/testPict2.jpg"
#     file = "Images/tinyTest.jpg"
#     file = "Images/ellipse.jpg"
    file = "Images/FiberImages/30_LGF/LGF_EGP_30wt__1sec_79Deg_xz-plane_F21_0_W_10_L_20X_R(color).jpg"
#     file = "TestCases/adjoined/brown12.jpg"
#     file = "TestCases/watershedTest2.bmp"
    minW = 15
    try:
        im2, out, outlines = standAlone(file, minW)
    except TypeError:
        raise
        print("standAlone died.", )
        
        d2 = datetime.datetime.now()
        diff = d2-d1
        print("Time elapsed before death:", diff)
        return
    
    out.show()
    im2.show()
    # im2.save("output.bmp")
    # out.save("output2.bmp")
    
    d2 = datetime.datetime.now()
    diff = d2-d1
    print("Time elapsed:", diff)

if __name__ == "__main__":
    import cProfile
    cProfile.run('main()')
#     main()
    