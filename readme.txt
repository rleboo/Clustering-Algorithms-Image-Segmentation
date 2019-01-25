Raymond Leboo

Files Included:
kmeans.py
seg.py
Earth.gif
rainbow.gif
venus_fly_trap.gif
Earth (folder of images)
rainbow (folder of images)
venus (folder of images)


This project implements different clustering algorithms to perform image segmentation. It uses the partition algorithm of K-means and the agglomerative algorithms of UPGMA and NJ to carry out image segmentation (seperately). The kmeans.py code was written by Stephen Marsland, 2008, 2014. 

--seg.py--
To run: python3 seg.py <image file> <0 for UPGMA. 1 for NJ>
Output: On the terminal it asks the user to input a number of desired clusters upto a maximum. Outputs a window with three images side by side. The original image on the left, either UPGMA or NJ in the middle, and K-means on the right. In the terminal it outputs the time taken to carry out UPGMA/NJ and K-means. 
