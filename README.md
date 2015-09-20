# FiberFitting
Processes an image of cross-cut fibers, fitting an ellipse to each fiber cross-section.

When finished, the program will:
  go through all the images in the current working directory
  for each image, it will:
    check a large number of equally spaced points on the image, to find points of high contrast along a line
    for each point it finds, it will:
      look for another point of high contrast tangentially near the previous one, moving CCW.
      Repeat until the entire fiber outline is traced
      After finding the full outline of a fiber, it will:
        break it into 2 or more sub-lists at all points of high curvature, if there are any such points.
        approximate and save the parameters of the ellipse which best fits the given outline.
        Color in the area contained by that outline on the input image.
    Once all the fibers have been traced and approximated, a binary image is created on which all the approximate ellipses are drawn (in white), and the parameters of all those ellipses are saved to a file.
    That file can be loaded into a gui, which will generate and display a black and white image of the ellipses which the user can edit by clicking an ellipse and pressing keyboard keys to move it, rotate it, stretch it, or delete it. New ellipses can also be added.
    The gui will let you save your edited version as a new file or binary image.
