from PIL import Image, ImageDraw
import math

# what follows will display a graph of the amount of each type of color in the picture
def drawData(graph, colorCountList, divs, numPixels, col):
    dataSet = []
    for i in range(0, 256):
        graph.putpixel( (i, 30), (0,0,0))
        x = i
        y = 30 + int(256 * colorCountList[i/divs][col] / float(numPixels))
        dataSet.append((x,y))
        if ( i > 0 ):
            draw = ImageDraw.Draw(graph)
            draw.line([(dataSet[i-1][0],dataSet[i-1][1]), (dataSet[i][0],dataSet[i][1])],
                       (255*int(col == 0),255*int(col == 1),255*int(col == 2)), 1)

def graphData(colorCountArray, divs, numPixels):
    graph = Image.new("RGB", (256, 300), "black")
    for i in range(0, 256):
        for j in range(0, 10):
            graph.putpixel( (i, j), (i,0,0))
            graph.putpixel( (i, j+10), (0,i,0))
            graph.putpixel( (i, j+20), (0,0,i))
        if (i % divs == 0) & False:
            for c in range(30, 300):
                graph.putpixel( (i, c), (255, 255, 255))
    for i in range(0,3):
        drawData( graph, colorCountArray, divs, numPixels, i )
    graph.show()

def countColors( pixels, width, height, divs):
    
    # initialize an array with all zeros
    colorCountArray = []
    for i in range(0, int(256/divs + 1)):
        colorCountArray.append([0,0,0])
         
    for i in range(0,3):
        #increment each piece of the array when a color fits into that group
        for j in range(0, width):
            for k in range(0, height):
                colorCountArray[ (int(pixels[j,k][i] / divs)) ][i] += 1
                
    # uncomment this to display a graph of the colors in an image
#     graphData(colorCountArray, divs, width * height)
    return colorCountArray
 
def getStats( pixels, width, height ):
    divs = 3
    colorArray = countColors( pixels, width, height, divs)
    avg = [0,0,0]
    stdev = [0,0,0]
    for i in range(0,3):
        for j in range(0, int(256/divs)):
            avg[i] += colorArray[j][i] * j * divs
        avg[i] /= (width * height)

    print("average found.")    
    for i in range(0,3):
        for j in range(0, int(256/divs)):
            # for each subdivision of color, calculate the stdev in the image
            for k in range(0,colorArray[j][i]):
                stdev[i] += pow(j * divs - avg[i], 2)
        stdev[i] /= (width*height)
        stdev[i] = math.sqrt( stdev[i] )

    print("stdevs found.")
      
    return(stdev, avg)


