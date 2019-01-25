Raymond Leboo
CPS 361

Files Included:
kmeans.py
seg.py
Earth.gif
rainbow.gif
venus_fly_trap.gif
Earth (folder of images)
rainbow (folder of images)
venus (folder of images)


This project implements different clustering algorithms to perform image segmentation. It uses the partition algorithm of K-means and the agglomerative algorithms of UPGMA and NJ to carry out image segmentation (seperately).

--seg.py--
To run: python3 seg.py <image file> <0 for UPGMA. 1 for NJ>
Output: On the terminal it asks the user to input a number of desired clusters upto a maximum. Outputs a window with three images side by side. The original image on the left, either UPGMA or NJ in the middle, and K-means on the right. In the terminal it outputs the time take for UPGMA/NJ and K-means. 
