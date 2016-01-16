'''
Created on Jul 29, 2015

@author: james

I got the poly_oval function directly from "Stephen D Evans" at
https://mail.python.org/pipermail/python-list/2000-December/022013.html

Creates a GUI for easier and more accurate editing of ellipse to an image of fibers.

User actions:

    input a color image, process it via the method of highest contrasts, and then open the result
    
    open a saved file containing the result of a previous run
     
    input a color image as well as a partial binary image, to be corrected using this GUI 

Buttons:

    Use mousewheel to scroll vertically in the image
    
    Right-click on the image to place a new ellipse there.
        The ellipse starts as a circle of radius r = minimum allowable minor axis length
        
    There are superfluous 'Delete' and 'Undo' buttons in case someone prefers them.
    
    "Show Options" brings up an options window, in which you can change:
        -The minimum minor axis length of an ellipse
        -Any of the key bindings
    
    "Save" saves your work to a text file, to be resumed later.
        save format is originalFileName_editFile.txt
    
    "Create Binary" creates a binary image using the locations and parameters of all the existing ellipses.
        It saves it as originalFileName_BW_(PyFixed).jpg
        to both match and differentiate itself from the matlab code's naming conventions.
    
default key bindings:

    decrease major axis   = 'q'
    increase major axis   = 'e'
    decrease minor axis   = 'z'
    increase minor axis   = 'c'
    move ellipse left     = 'a'
    move ellipse right    = 'd'
    move ellipse up       = 'w'
    move ellipse down     = 's'
    rotate ellipse CW     = '3'
    rotate ellipse CCW    = '1'
    delete ellipse        = '2'
    undo last deletion    = 'x'
    
    hold to scroll horizontally using mousewheel = 'Shift_L'
    
    hold to zoom in/out using mousewheel = 'Control_L'  NOT IMPLIMENTED
    
    toggle fast change on/off = 'Alt_L'
        means that movements and changes of the selected ellipse happen much faster
        turn on for large, coarse adjustments
        I haven't really found it necessary, but if someone got the hang of the program it'd probably 
        save them some time.
    
    hide the ellipses while pressed, so you can better see the image  = 'space'

'''
from main import standAlone, fillEllipse
from math import pi, sqrt, cos, sin
import tkinter as tk
from tkinter import *
from PIL import Image
import os
import datetime

def poly_oval(x0,y0, x1,y1, steps=20, rotation=0):
    """return an oval as coordinates suitable for create_polygon"""

    # x0,y0,x1,y1 are as create_oval

    # rotation is in degrees anti-clockwise, convert to radians
    rotation = rotation * pi / 180.0

    # major and minor axes
    a = (x1 - x0) / 2.0
    b = (y1 - y0) / 2.0

    # center
    xc = x0 + a
    yc = y0 + b

    point_list = []

    # create the oval as a list of points
    for i in range(steps):

        # Calculate the angle for this step
        # 360 degrees == 2 pi radians
        theta = (pi * 2) * (float(i) / steps)

        x1 = a * cos(theta)
        y1 = b * sin(theta)

        # rotate x, y
        x = (x1 * cos(rotation)) + (y1 * sin(rotation))
        y = (y1 * cos(rotation)) - (x1 * sin(rotation))

        point_list.append(round(x + xc))
        point_list.append(round(y + yc))

    return point_list

#t is the distance in radians from the x axis, CCW, to the major axis.


class Ellipse:
    a = 0
    b = 0
    h = 0
    k = 0
    t = 0
    index = 0
    polygon = 0
    def __init__(self, a, b, h, k, t, i, polygon):
        self.a = a
        self.b = b
        self.h = h
        self.k = k
        self.t = t
        self.polygon = polygon
        self.index = i
    
    def params(self):
        return self.a, self.b, self.h, self.k, self.t
    
    def clicked(self, canvas):
            canvas.itemconfigure(self.polygon, tags = ("selected"))
            for poly in canvas.find_withtag("selected"):
                canvas.itemconfigure(poly, outline='')
            canvas.itemconfigure(self.polygon, outline='red')
            global slctdIndx
            slctdIndx = self.index
            
    def bindMe(self, canvas):
        canvas.tag_bind(self.polygon, 
                        '<ButtonPress-1>', 
                         lambda _: self.clicked(canvas), add = False)
        '''
        add binding to key presses to change parameters of ellipse
        '''

class EllipseCollection:
    ellipseList = []
    removedEllipses = []
    
    wdth = 0
    hght = 0
    
    root = 0
    canvas = 0
    menu = 0
    optionMenu = 0
    background = 0
    im = 0
    imName = ''
    
    scrollHorz = False
    scrollZoom = False
    moveFast  =False
    scale = 1
    minEllW = 10
    horzScrollCount = 0
    vertScrollCount = 0
    
    keyMappings = []
    keyMappingsDescript = []
    
    def __init__(self, w,h, elList, minW, name, keys = []):
        self.minEllW = minW
        self.ellipseList = []
        self.removedEllipses = keys
        self.wdth = w
        self.hght = h
        self.vertScrollCount = 0
        self.horzScrollCount = 0
        self.root = Tk()
        self.root.wm_title("Gooey GUI")
        self.menu = 0
        self.optionMenu = 0
        self.keyMappings = keys
        self.canvas = Canvas(self.root, width=self.wdth, height=self.hght)
#         self.background = self.canvas.create_rectangle((0, 0, self.wdth, self.hght), fill = "black")
        self.imName = name
        self.im = PhotoImage(file = name)
        self.background = self.canvas.create_image(w/2,h/2,image=self.im)
        
#         self.canvas.config(scrollregion=self.canvas.bbox(ALL))
#         self.scale = 1.0
#         self.background = Image.open(File)
#         self.img = None
#         self.img_id = None
#         # draw the initial image at 1x scale
#         self.redraw()

        self.setKeys()
        
        self.setKeyDescripts()
        
        for i in range( 0, len(elList)):
            h, k, t, a, b = elList[i]
#             print("load",i, a, b, h, k, t)
            ellipse = self.drawEllipse(a, b, h, k, t, i)
            self.ellipseList.append(ellipse)
#             print(ellipse)
#             print(self.ellipseList[i])
            self.ellipseList[i].bindMe(self.canvas)
        
        self.menu = Toplevel()
        self.menu.wm_title("Menu")
        
        self.makeButtons()
        
        self.bindKeys()
        
        self.root.mainloop()

    def makeButtons(self):
        w = 20
        delete = Button(self.menu, text = "Delete", command = self.deleteSelection, anchor = W)
        delete.configure(width = w, activebackground = "#33B5E5", relief = FLAT)
        delete.pack()
        
        undo = Button(self.menu, text = "Undo", command = self.undoDelete, anchor = W)
        undo.configure(width = w, activebackground = "#33B5E5", relief = FLAT)
        undo.pack()
        
        showOptions = Button(self.menu, text = "Options", command = self.shwOptns, anchor = W)
        showOptions.configure(width = w, activebackground = "#33B5E5", relief = FLAT)
        showOptions.pack()
        
        save = Button(self.menu, text = "Save", command = self.saveData, anchor = W)
        save.configure(width = w, activebackground = "#33B5E5", relief = FLAT)
        save.pack()
        
        makeImage = Button(self.menu, text = "Create Binary", command = self.makeImage, anchor = W)
        makeImage.configure(width = w, activebackground = "#33B5E5", relief = FLAT)
        makeImage.pack()
        
        showBinds = Button(self.menu, text = "Show Key Mapping", command = self.showKeyBindings, anchor = W)
        showBinds.configure(width = w, activebackground = "#33B5E5", relief = FLAT)
        showBinds.pack()
        
        self.canvas.pack(side=TOP, expand=True, fill=BOTH)
    
    def bindKeys(self):
        self.canvas.bind_all('<KeyPress>', self.keyPressed)
        self.canvas.bind_all('<KeyRelease-Shift_L>', self.stopScrollHz)
        self.canvas.bind_all('<KeyRelease-Control_L>', self.stopScrollZm)
        self.canvas.bind_all('<KeyRelease-space>', self.sendBack)
        
        self.canvas.bind("<Button-3>", self.addNewEllipse)
        
        self.canvas.bind_all('<Button-4>', self.mouseWheelUp)
        self.canvas.bind_all('<Button-5>', self.mouseWheelDown)
    
    def setKeyDescripts(self):
        self.keyMappingsDescript.append("Decrease major axis")
        self.keyMappingsDescript.append("Increase major axis")
        self.keyMappingsDescript.append("Decrease minor axis")
        self.keyMappingsDescript.append("Increase minor axis")
        self.keyMappingsDescript.append("Move ellipse left")
        self.keyMappingsDescript.append("Move ellipse right")
        self.keyMappingsDescript.append("Move ellipse up")
        self.keyMappingsDescript.append("Move ellipse down")
        self.keyMappingsDescript.append("Rotate ellipse CW")
        self.keyMappingsDescript.append("Rotate ellipse CCW")
        self.keyMappingsDescript.append("Delete ellipse")
        self.keyMappingsDescript.append("Undo last deletion")
        self.keyMappingsDescript.append("Hold to scroll horizontally (wheel)")
        self.keyMappingsDescript.append("Hold to zoom in/out (NOT IMPLIMENTED)")
        self.keyMappingsDescript.append("Toggle coarse adjustment mode")
        self.keyMappingsDescript.append("Hold to temporarily hide ellipses")

    def shwOptns(self):
        self.optionMenu = Toplevel()
        self.optionMenu.wm_title("Options")
        e = Entry(self.optionMenu)
        e.pack()
        def setMinW():
            try:
                self.minEllW = float(e.get())
            except Exception:
                self.minEllW = 10
            finally:
                e.delete(0, END)
        setVal = Button(self.optionMenu, text = ("Set minimum ellipse width, currently %",self.minEllW), 
                        command = setMinW, anchor = W)
        setVal.configure(width = 20, activebackground = "#33B5E5", relief = FLAT)
        setVal.pack()
        
        e.delete(0, END)
        
#         print("Other options not yet implemented")

#         OPTIONS = self.keyMappingsDescript
#         
#         variable = StringVar(self.root)
#         variable.set(OPTIONS[0]) # default value
#         
#         w = OptionMenu(self.optionMenu, variable, OPTIONS[:])
#         w.pack()
    
    def showKeyBindings(self):
        showKeys = Toplevel()
        showKeys.wm_title("Current Key Mapping")
        
        longest = 0
        for i in range(len(self.keyMappingsDescript)):
            if len(self.keyMappingsDescript[i]) > longest:
                longest = len(self.keyMappingsDescript[i])
        descr = Text(showKeys, height=len(self.keyMappingsDescript), width=longest+3)
        descr.pack(side=LEFT)
        
        key = Text(showKeys, height=len(self.keyMappingsDescript), width=10)
        key.pack(side=RIGHT)
        
        for i in range(len(self.keyMappingsDescript)):
            string = self.keyMappingsDescript[i]
            val = self.keyMappings[i]
            descr.insert("end", string+"\n")
            key.insert("end", val+"\n")
        
    
    def makeImage(self):
        final = Image.new('L', (self.wdth,self.hght), 0)
        for i in range(len(self.ellipseList)):
            a, b, x0, y0, alpha = self.ellipseList[i].params()
            fillEllipse(final, x0, y0, alpha, a, b, 255)
        final.show()
        for i in range(len(self.imName)):
            if self.imName[i] == '.':
                break
        fileName = self.imName[len(self.imName)-i:]+"_BW_(PyFixed).jpg"
        final.save(fileName)
        
#     def zoom(self,event):
#         if event.num == 4:
#             self.scale *= 5/4
#         elif event.num == 5:
#             self.scale *= 4/5
#         self.redraw(event.x, event.y)
#     
#     def redraw(self, x=0, y=0):
#         if self.img_id: self.canvas.delete(self.img_id)
#         iw, ih = self.orig_img.size
#         # calculate crop rect
#         cw, ch = iw / self.scale, ih / self.scale
#         if cw > iw or ch > ih:
#             cw = iw
#             ch = ih
#         # crop it
#         _x = int(iw/2 - cw/2)
#         _y = int(ih/2 - ch/2)
#         tmp = self.orig_img.crop((_x, _y, _x + int(cw), _y + int(ch)))
#         size = int(cw * self.scale), int(ch * self.scale)
#         # draw
#         self.img = ImageTk.PhotoImage(tmp.resize(size))
#         self.img_id = self.canvas.create_image(x, y, image=self.img)
#         gc.collect()
        
    def mouseWheelUp(self, event):
        print(self.vertScrollCount)
        if self.scrollHorz:
            self.canvas.xview_scroll(-1, "units")
            self.horzScrollCount -= 1
            
        elif self.scrollZoom:
            ()
#             self.canvas.scale(ALL, event.x, event.y, 5/4, 5/4)
#             self.zoom(event)
        else:
            self.canvas.yview_scroll(-1, "units")
            self.vertScrollCount -= 1
    
    def mouseWheelDown(self, event):

        if self.scrollHorz:
            self.canvas.xview_scroll(+1, "units")
            self.horzScrollCount += 1
            
        elif self.scrollZoom:
            ()
#             self.canvas.scale(ALL, event.x, event.y, 4/5, 4/5)
#             self.zoom(event)
        else:
            self.canvas.yview_scroll(1, "units")
            self.vertScrollCount = 1
    
    def stopScrollHz(self, event):
        self.scrollHorz = False
        
    def stopScrollZm(self, event):
        self.scrollZoom = False
        
    def sendBack(self, event):
        self.canvas.tag_lower(self.background)
        
#     def setToDrawOnClick(self):
#         self.addOnClick = True
        
    def drawEllipse( self, a, b, h, k, t, i, event = 0 ):
        #these variables are for the pre-rotated ellipse
#         x,y = 0,0
        if event != 0:
            h = self.canvas.canvasx(event.x)
            k = self.canvas.canvasy(event.y)

        xStart = h - a
        yStart = k - b
        xEnd = h + a
        yEnd = k + b
        resolution = 2 # 1 is the smallest. It represents the number of gaps between points
        steps = int(2*pi*sqrt((a**2 + b**2) / 2) / resolution)
        polygon = self.canvas.create_polygon(
                tuple(poly_oval(xStart, yStart, xEnd, yEnd, steps, rotation = -t/pi * 180)),
                fill = "white",
                stipple="gray50")
        ellipse = Ellipse(a, b, h, k, t, i, polygon)
        ellipse.clicked(self.canvas)
        # use a polygon to draw an oval rotated 30 degrees anti-clockwise
        return ellipse
    
    
    def deleteSelection(self):
        global slctdIndx
        if slctdIndx != -10:
            self.removedEllipses.append(self.ellipseList[slctdIndx])
            self.canvas.delete(self.ellipseList[slctdIndx].polygon)
#             del self.ellipseList[slctdIndx]
            self.resetSelection()
        else:
            print("No ellipse selected")
    
    def undoDelete(self):
        if len(self.removedEllipses) > 0:
#             ell = self.removedEllipses[len(self.removedEllipses) - 1]
            ell = self.removedEllipses.pop()
            a = ell.a
            b = ell.b
            h = ell.h
            k = ell.k
            t = ell.t
            i = ell.index
            ell = self.drawEllipse(a, b, h, k, t, i)
            self.ellipseList[i] = ell
            ell.bindMe(self.canvas)
#             print(a,b,h,k,t, i, len(self.ellipseList))
        else:
            print("No ellipses have been deleted.")
    
    def resetSelection(self):
        for i in range(0, len(self.ellipseList)):
            self.canvas.itemconfigure(self.ellipseList[i].polygon, outline='')
        global slctdIndx
        slctdIndx = -10
        
    def addNewEllipse(self, event):
        
        #p is the center point
        p = (event.x, event.y)
        newEll = self.drawEllipse( self.minEllW, self.minEllW, p[0], p[1], 0, len(self.ellipseList), event )
        self.ellipseList.append(newEll)
        self.ellipseList[len(self.ellipseList)-1].bindMe(self.canvas)
        
    def setKeys(self, newKeys = []):
        #if the keys aren't mapped or something went wrong with the file
        if len(newKeys) != 16:
            self.keyMappings = []
            for i in range(16):
                self.keyMappings.append('')
            self.keyMappings[0] = 'q'
            self.keyMappings[1] = 'e'
            self.keyMappings[2] = 'z'
            self.keyMappings[3] = 'c'
            self.keyMappings[4] = 'a'
            self.keyMappings[5] = 'd'
            self.keyMappings[6] = 'w'
            self.keyMappings[7] = 's'
            self.keyMappings[8] = '3'
            self.keyMappings[9] = '1'
            self.keyMappings[10] = '2'
            self.keyMappings[11] = 'x'
            self.keyMappings[12] = 'Shift_L'
            self.keyMappings[13] = 'Control_L'
            self.keyMappings[14] = 'Alt_L'
            self.keyMappings[15] = 'space'
        else:
            self.keyMappings = newKeys
            
    def keyPressed(self, event):
        global slctdIndx
        # lower case shrinks that value, upper increases it.
        a = self.keyMappings[0]
        A = self.keyMappings[1]
        b = self.keyMappings[2]
        B = self.keyMappings[3]
        h = self.keyMappings[4]
        H = self.keyMappings[5]
        k = self.keyMappings[6]
        K = self.keyMappings[7]
        t = self.keyMappings[8]
        T = self.keyMappings[9]
        rmv = self.keyMappings[10]
        undo = self.keyMappings[11]
        scrlHz = self.keyMappings[12]
        scrlZm = self.keyMappings[13]
        fastMv = self.keyMappings[14]
        hideElls = self.keyMappings[15]
        print(event)
        if event.keysym == scrlHz:
            self.scrollHorz = True
            return
        elif event.keysym == scrlZm:
            self.scrollZoom = True
            return
        elif event.keysym == fastMv:
            self.moveFast = not self.moveFast
            return
        elif event.keysym == hideElls:
            self.canvas.tag_raise(self.background)
            return
        delta = 1
        if self.moveFast:
            delta *= 5
        if event.keysym == undo:
                self.undoDelete()
        elif slctdIndx != -10:
            changeSeen = False

            if event.keysym == H:
                if self.ellipseList[slctdIndx].h < self.wdth - delta:
                    self.ellipseList[slctdIndx].h += delta
                    changeSeen = True
#                     print(H)
            elif event.keysym == h:
                if self.ellipseList[slctdIndx].h > delta-1:
                    self.ellipseList[slctdIndx].h -= delta
                    changeSeen = True
#                     print(h)
            elif event.keysym == k:
                if self.ellipseList[slctdIndx].k > delta-1:
                    self.ellipseList[slctdIndx].k -= delta
                    changeSeen = True
#                     print(k)
            elif event.keysym == K:
                if self.ellipseList[slctdIndx].k < self.hght - delta:
                    self.ellipseList[slctdIndx].k += delta
                    changeSeen = True
#                     print(K)
            elif event.keysym == a:
                if self.ellipseList[slctdIndx].a > self.ellipseList[slctdIndx].b:
                    self.ellipseList[slctdIndx].a -= delta
                    changeSeen = True
#                     print(a)
            elif event.keysym == A:
                if self.ellipseList[slctdIndx].a < self.hght:
                    self.ellipseList[slctdIndx].a += delta
                    changeSeen = True
#                     print(A)
            elif event.keysym == b:
                if self.ellipseList[slctdIndx].b > self.minEllW:
                    self.ellipseList[slctdIndx].b -= delta
                    changeSeen = True
#                     print(b)
            elif event.keysym == B:
                if self.ellipseList[slctdIndx].b < self.ellipseList[slctdIndx].a:
                    self.ellipseList[slctdIndx].b += delta
                    changeSeen = True
#                     print(B)
            elif event.keysym == t:
                angle = self.ellipseList[slctdIndx].t / pi * 180
                angle -= delta
                if angle < 0:
                    angle = 180 - angle
                self.ellipseList[slctdIndx].t = angle/180 * pi
                changeSeen = True
#                 print(t)
            elif event.keysym == T:
                angle = self.ellipseList[slctdIndx].t / pi * 180
                angle += delta
                if angle > 180:
                    angle = 0 + angle
                self.ellipseList[slctdIndx].t = angle/180 * pi
                changeSeen = True
#                 print(T)
            elif event.keysym == rmv:
                self.deleteSelection()
                
            if changeSeen:
                # redraw the ellipse
                temp = slctdIndx
                self.deleteSelection()
                self.undoDelete()
                slctdIndx = temp

        else:
            print("No ellipse selected.")
    
    def saveData(self):
        for i in range(len(self.imName)):
            if self.imName[i] == '.':
                break

        fileName = self.imName[:i] + "_editFile.txt"
        folderName = "savedEllipseData"
        os.mkdir( folderName )
        f = open(os.path.join(folderName, fileName), 'w')
        
#         f.write(self.imName)
        f.write(self.wdth, "\t", self.hght)
        for i in range(len(self.ellipseList)):
            f.write(self.ellipseList.a, self.ellipseList.b, self.ellipseList.x0, 
                    self.ellipseList.y0, self.ellipseList.alpha)
        

def loadData( fileName ):
    f = open(fileName, 'w')
    
    minW = 20
    
    return ("Images/tinyTest",".jpg"), 10, [], []



# list1 = []
# list1.append((100, 50, 200, 200, pi/8))
# list1.append((20, 20, 45, 90, pi/8))
# list1.append((20, 20, 145, 90, pi/8))
# list1.append((20, 20, 45, 190, pi/8))
# list1.append((80, 20, 20, 0, 3*pi/4))
def main():
    '''
    add display of current minW value to the changer thingy
    '''
    
    name, minW, keys, ellipses = loadData("dataFile.txt")
    # name = "Images/testPictTiny",".jpg"
    # name = "Images/FiberImages/44.5_LCF/LCF_EGP_44.5wt__2sec_79Deg_xz-plane_C9_0_W_10_L_50x_~1.5mm_Fixed",".jpg"
    
    # im = Image.open(name)
    name = "Images/tinyTest.jpg"
    name = "Images/ellipse.jpg"
    j1,j2, ellipses = standAlone(name, minW)
    # j1.show()
    # j2.show()
    # j1.save()
    # j2.save()
    # ellipses = []
    name = ("Images/ellipse",".jpg")
    im = Image.open(name[0]+name[1])
    name = name[0] + ".gif"
    im.save(name)
    w, h = im.size
    
    slctdIndx = -10
    
    # os.system('xset r off')
    gui = EllipseCollection( w, h, ellipses, minW, name, keys )
    # os.system('xset r on')


if __name__ == "__main__":
    
    main()


