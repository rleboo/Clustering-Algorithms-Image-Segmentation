import sys
import cImage
import math
import kmeans
import numpy as np
import copy
import time


def neighbourJoining(kme, cluster, matrix, fincl):
    
    # Used for testing the accuracy of the implementation
    """
    l1 = [0.0, 5.0, 4.0, 7.0, 6.0, 8.0]
    l2 = [5.0, 0.0, 7.0, 10.0, 9.0, 11.0]
    l3 = [4.0, 7.0, 0.0, 7.0, 6.0, 8.0]
    l4 = [7.0, 10.0, 7.0, 0.0, 5.0, 9.0]
    l5 = [6.0, 9.0, 6.0, 5.0, 0.0, 8.0]
    l6 = [8.0, 11.0, 8.0, 9.0, 8.0, 0.0]

    arr = np.vstack((l1, l2))
    arr = np.vstack((arr, l3))
    arr = np.vstack((arr, l4))
    arr = np.vstack((arr, l5))
    arr = np.vstack((arr, l6))

    arr.astype(float)    
    matrix = arr
    cluster = [[i] for i in range(len(matrix))]
    fincl = [[i] for i in range(len(matrix))]
    """

    numcluster = len(matrix)
    outF = copy.deepcopy(matrix)

    # Loops until all clusters are equal to user inputed cluster
    while kme < numcluster:
        tempList = [0] * len(matrix)
        # Calculate the almost average
        for i in range(len(matrix)):
            # Checks if list is all math.inf
            if all(x == math.inf for x in matrix[i]):
                outF[i] = math.inf
                tempList[i] = math.inf
            else:
                stotal = 0
                for x in range(len(matrix[i])):
                    if(matrix[i][x] == math.inf):
                        continue
                    elif i == x:
                        continue
                    else:
                        stotal += matrix[i][x] 
                tempList[i] = stotal
                #tempList[i] = (sum(matrix[i]) - matrix[i][0])/(numcluster - 2)
        
        # Calculate the M(ij) -  u(i) - u(j) --> Need to find smallest
        for x in range(len(matrix)):
            for y in range(len(matrix)):
                if x == y:
                    outF[x][y] = math.inf
                elif matrix[x][y] == math.inf:
                    outF[x][y] = math.inf
                else:
                    # Almost distance
                    if(numcluster == 2):
                        outF[x][y] = matrix[x][y] - (tempList[x] - matrix[x][y])/(1) - (tempList[y]- matrix[x][y])/(1)
                    else:
                        outF[x][y] = matrix[x][y] - (tempList[x] - matrix[x][y])/(numcluster - 2) - (tempList[y]- matrix[x][y])/(numcluster - 2)

        first = math.inf
        f1 = 0
        f0 = 0
        # Finds the smallest pair
        for x in range(len(matrix)):
            for y in range(x):
                if outF[x][y] < first:
                    first = outF[x][y]
                    f1 = y
                    f0 = x
        copym = copy.deepcopy(matrix)
        # Clustering together
        if len(cluster[f1]) <= len(cluster[f0]):
            # Set rows/col of cluster to be merged to inf
            matrix[:, f0] = math.inf
            matrix[f0,:] = math.inf
            # Merged cluster
            fincl[f1] = cluster[f1] + cluster[f0]
            cluster[f1] = cluster[f1] + cluster[f0]
            # Other cluster points to index of new cluster
            fincl[f0] = [f1]
            # index thats going to contain recaluclated value
            finc = f1
            # index of merged clusters
            ot = f0

        else:
             # Set rows/col of cluster to be merged to inf
            matrix[:, f1] = math.inf
            matrix[f1,:] = math.inf
             # Merged cluster
            fincl[f0] = cluster[f1] + cluster[f0]
            cluster[f0] = cluster[f0] + cluster[f1]
            # Other cluster points to index of new cluster
            fincl[f1] = [f0]
            # index thats going to contain recaluclated value
            finc = f0
             # index of merged clusters
            ot = f1

        # Computing distance between new clusters and all other clusters
        # M(ij)k = (M(ik) + M(jk) - M(ij)) / 2
        for i in range(len(cluster)):
            if i == finc:
                matrix[finc][i] = 0.0
            elif matrix[finc][i] == math.inf:
                continue
            else:
                # M(ij)k = (M(ik) + M(jk) - M(ij)) / 2
                total = (copym[i][finc] + copym[i][ot] - copym[finc][ot])/2
                matrix[finc][i] = total
                matrix[i][finc] = total
        numcluster -= 1
    return fincl


def upgma(kme, cluster, matrix, fincl):
    # Used to test accuracy of Upgma 
    
    """
    l1 = [0.0, 8.0, 7.0, 12.0]
    l2 = [8.0, 0.0, 9.0, 14.0]
    l3 = [7.0, 9.0, 0.0 , 11.0]
    l4 = [12.0, 14.0, 11.0, 0.0]

    arr = np.vstack((l1, l2))
    arr = np.vstack((arr, l3))
    arr = np.vstack((arr, l4))
    arr.astype(float)    wholepixel = oldimage.getPixel(row, col)


    matrix = arr
    cluster = [[i] for i in range(len(matrix))]
    """


    numcluster = len(matrix)
    copym = copy.deepcopy(matrix)

    # Loops until all clusters are equal to user inputed cluster
    while kme < numcluster:
        first = sys.maxsize
        f1 = 0
        f0 = 0
        # Finding the smallest value in the distance matrix
        for x in range(len(fincl)):
            for y in range(x):
                if matrix[x][y] < first:
                    first = matrix[x][y]
                    f1 = y
                    f0 = x 
        finc = 0
        # Adding smaller cluster to bigger cluster
        if len(cluster[f1]) <= len(cluster[f0]):
            matrix[:, f0] = math.inf
            matrix[f0,:] = math.inf
            #Keeps track of whose in the cluster
            fincl[f1] = cluster[f1] + cluster[f0]
            cluster[f1] = cluster[f1] + cluster[f0]
            # Cluster f0 is at index f1
            fincl[f0] = [f1]
            finc = f1

        else:
            matrix[:, f1] = math.inf
            matrix[f1,:] = math.inf
            fincl[f0] = cluster[f1] + cluster[f0]
            # Cluster f1 is at index f0
            fincl[f1] = [f0]
            #Keeps track of whose in the cluster
            cluster[f0] = cluster[f0] + cluster[f1]
            finc = f0

        # recalculating the distance matrix
        for i in range(len(cluster)):
            if i == finc:
                matrix[finc][i] = math.inf
            elif matrix[finc][i] == math.inf:
                continue
            else:
                total = 0
                loop = 0
                # The original distance from the data points inside cluster to new position
                # Added together
                for x in cluster[finc]:
                    for y in cluster[i]:
                        total += copym[x][y]
                        loop += 1
                # Change both position in distance matrix
                matrix[finc][i] = total/loop
                matrix[i][finc] = total/loop

        numcluster -= 1
    return fincl


# Function maps out which pixel is in one of the kme clusters
def fitcluster(fincl, kme):
    clus = [0] * kme
    added = [False] * len(fincl)
    loop = 0
    clust = 0
    for i in range(len(fincl)):
        if(len(fincl[i]) != 1):
            # If it has already been clustered skip
            if(added[i] == True):
                continue
            else:
                # Add all the item in cluster to visited. Prevents repeats
                for item in fincl[i]:
                    added[item] = True
                clus[clust] = fincl[i]
                clust += 1
            loop += 1
        # If cluster has the same index and cluster. It hasn't been clustered
        elif i == fincl[i][0]:
            clus[clust] = fincl[i]
            added[i] = True
            loop += 1
            clust += 1
        else:
            # Else someone cluster this index and put their index. 
            # Finds final cluster index
            if(added[i] == True):
                continue
            else:
        
                previous = fincl[i][0]
                trackn = fincl[i]
                track = len(trackn)
                # Traces back who clusted them
                while(track == 1):
                    previous = trackn[0]
                    trackn = fincl[trackn[0]] 
                    track = len(trackn)
                    loop += 1
                if(added[trackn[0]] == False):
                    for item in trackn:
                        added[item] = True

                clus[clust] = trackn
                clust += 1
    return clus


def main():
    oldimage = cImage.FileImage(sys.argv[1])
    # Whether we use nj clustering or not
    nj = int(sys.argv[2])

    #Create window triple the width to store two images side by side
    myimagewindow = cImage.ImageWin("Image Window", oldimage.getWidth() * 3, oldimage.getHeight())

    #Set oldimage to top left corner
    oldimage.setPosition(0,0)
    # Draw oldiage

    oldimage.draw(myimagewindow)

    image = []
    kmeandata = []
    # Used later to reassign pixels
    straight = []
    # Get all pixels into a single array 
    for row in range(oldimage.getHeight()):
        colr = []
        for col in range(oldimage.getWidth()):
            pixel = []
            wholepixel = oldimage.getPixel(col, row)
            pixel.append(wholepixel.getRed())
            pixel.append(wholepixel.getGreen())
            pixel.append(wholepixel.getBlue())
            kmeandata.append(pixel)
            colr.append(pixel)
            if pixel not in straight:
                straight.append(pixel)
            else:
                continue
        image.append(colr)

    # Asks user how many clusters they can make 
    kme = int(input("How many clusters? You can use up-to " + str(len(straight)) + " clusters: "))

    finalar = np.array(straight)

    pidata = copy.deepcopy(finalar)


    cluster = [[i] for i in range(len(finalar))]

    li = [[] for i in range(len(finalar))]
    matrix = np.zeros((len(finalar), len(finalar)))

    # Calculates eucledian distance between pixels  
    for i in range(len(finalar)):
        for j in range(len(finalar)):
            if (i == j):
                li[i].append(0)
            else:
                x1 = finalar[i][0]
                x2 = finalar[i][1]
                x3 = finalar[i][2]
                y1 = finalar[j][0]
                y2 = finalar[j][1]
                y3 = finalar[j][2]
                distance = math.sqrt((x1 - y1)**2 + (x2 - y2) ** 2 + (x3 - y3) ** 2)
                matrix[i][j] = distance
 
    finalar =  matrix
    fincl = [[i] for i in range(len(finalar))]

    numcluster = len(matrix)

    # Performs either upgma or nj depending on command line
    # Returns the where each pixel is clustered 
    ctime = time.time()
    if nj == 1:
        fincl = neighbourJoining(kme, cluster, matrix, fincl)
    else:
        fincl = upgma(kme, cluster, matrix, fincl)

    
    # Fits the different pixel clusters into kme*clusters using data return from clustering functions (fincl)
    clus = fitcluster(fincl, kme)

    pidata = np.array(pidata)

    #Which pixel is assigned to what cluster
    assign = [0] * len(finalar)
    #What the center clusters are
    fincluste = [0] * kme
    # This finds the averages of the clustered data
    # Assigniment is in assign list
    # Cluster averages are in finclusters
    #print(clus)
    for i in range(len(clus)):
        total = 0
        a = np.zeros(shape=(len(clus[i]),3))
        loop = 0
        for item in clus[i]:
            if(assign[0] != 0):
                print("That shouldn't have happened")
            else:
                assign[item] = i
            a[loop] = pidata[item]
            loop += 1
        pixel = a.mean(axis=0)
        # Cluster average pixels
        fincluste[i] = pixel.tolist()
    fclustime = time.time() - ctime
    print(fclustime)


    newimage = cImage.EmptyImage(oldimage.getWidth(), oldimage.getHeight())


    data = np.array([np.array(xi) for xi in straight])
    # make a kmeans object 
    ktime = time.time()
    kobj = kmeans.kmeans(kme, data)

    #Centers of the data
    centers = kobj.kmeanstrain(data)

    #Which cluster each data point is associated with
    output = kobj.kmeansfwd(data)
    ftime = time.time() - ktime

    output.astype(int)
    output = output.tolist()
    kimage = cImage.EmptyImage(oldimage.getWidth(), oldimage.getHeight())

    ktrack = 0
    track = 0
    straight = []
    # This draws the new image
    for row in range(oldimage.getHeight()):
        for col in range(oldimage.getWidth()):
            pixel = []
            wholepixel = oldimage.getPixel(col, row)
            pixel.append(wholepixel.getRed())
            pixel.append(wholepixel.getGreen())
            pixel.append(wholepixel.getBlue())
            if pixel not in straight:
                straight.append(pixel)
                num = int(assign[track])
                knum = int(sum(output[track]))
                track += 1

            else:
                value = straight.index(pixel)
                num =  int(assign[value])
                knum = int(sum(output[value]))

            rbgList = fincluste[num]
            newimage.setPixel(col, row, cImage.Pixel(int(rbgList[0]), int(rbgList[1]), int(rbgList[2])))

            #num = int(sum(output[ktrack]))
            krbg = centers[knum]
            rgb = tuple(krbg.tolist())
            # setting pixel to image
            kimage.setPixel(col, row, cImage.Pixel(int(rgb[0]), int(rgb[1]), int(rgb[2])))
            ktrack += 1

            
    kimage.setPosition(oldimage.getWidth() * 2 + 1, 0)
    # Set newimage right next to old image
    newimage.setPosition(oldimage.getWidth() + 1, 0)
    # Draw image on imagewindow
    newimage.draw(myimagewindow)
    kimage.draw(myimagewindow)
    # Exit image on click
    print(ftime)
    myimagewindow.exitOnClick()
main()
