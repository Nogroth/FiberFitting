# FiberFitting
Processes an image of cross-cut fibers, fitting an ellipse to each fiber cross-section.

When finished, the program will:

----Go through all the images in the current working directory
  
----For each image, it will:

----|----Check a large number of equally spaced points on the image, to find points of high contrast along a line

----|----For each point it finds, it will:

----|----|----Look for another point of high contrast tangentially near the previous one, moving CCW.

----|----|----Repeat until the entire fiber outline is traced

----|----|----After finding the full outline of a fiber, it will:

----|----|----|----Break it into 2 or more sub-lists at all points where two fibers intersect or where one fiber is  broken, if there are any such points.

----|----|----|----|----Approximate and save the parameters of the ellipse which best fits the given outline.

----|----|----|----|----Color in the area contained by that outline on the input image.

----|----|----|----If any 2 of those sub-lists are calculated to describe approximately equivalent ellipses, then only the average of the two ellipses is kept.

----|----Once all the fibers have been traced and approximated, a binary image is created on which all the approximate ellipses are drawn (in white), and the parameters of all those ellipses are saved to a file.

----|----That file can be loaded into a gui, which will generate and display a black and white image of the ellipses which the user can edit by clicking an ellipse and pressing keyboard keys to move it, rotate it, stretch it, or delete it. New ellipses can also be added.

----|----The gui will let you save your edited version as a new file or binary image.


Current shortcomings:

----It only processes the image located at the path written out in the code, not all the images in the cwd

----The splitting algorithm doesn't work very well, and has been turned off.

----The code to combine similar ellipses hasn't yet been written.

