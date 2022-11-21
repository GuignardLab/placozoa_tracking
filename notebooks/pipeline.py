# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:22:36 2022

@author: nessl
"""
# Packages

from tifffile import imread,imwrite
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import exposure,img_as_ubyte,morphology,filters
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from skimage.measure import label, regionprops_table


# Preprocessing

def stretch(im):
    '''
    Function to stretch the contrast of an image

    Parameters
    ----------
    im : image 
        raw image.

    Returns
    -------
    stretch : image
        image with stretched contrast for each planes.

    '''
    stretch = np.zeros_like(im)

    for i in range(np.shape(im)[0]):
        # Rescaling
        img = im[i,...].copy()
        p2, p98 = np.percentile(img, (2, 98))
        stretch[i,...] = exposure.rescale_intensity(img, in_range=(p2, p98))
        
    return stretch

def alg(im):
    '''
    Parameters
    ----------
    im : image
        the contrast-stretched image.

    Returns
    -------
    im_final : image
        the algae present in each frames.
    Function to get the algae per plane:
    workflow:
        - segment
        - remove largest component (the organism)
        - return the segmented image 

    '''
    
    im_final = np.zeros_like(im)
    
    # Preallocating mask array
    footprint = morphology.footprints.disk(1)
        
    for plane in range(np.shape(im)[0]):
        
        #print(f'Segmenting plane {plane} out of {np.shape(im)[0]}')
        print(f'{(plane/np.shape(im)[0])*100:.2f} % done ...')
        
        # Use white_tophat and gaussian filter on the frame
        image = im[plane,...]

        image = morphology.white_tophat(image, footprint)
        image = filters.gaussian(image, sigma = 2)

        # Threshold the result & label each detected region
        thresholds = filters.threshold_multiotsu(image)
        regions = np.digitize(image, bins=thresholds)
        label_image = regions

        # Fill the holes in the cell morphology
        fill = ndimage.binary_fill_holes(label_image)
        
        labels, num_features = ndimage.label(fill)
        label_unique = np.unique(labels)

        #count pixels of each component and sort them by size, excluding the background
        vol_list = []
        for labl in label_unique:
            if labl != 0:
                vol_list.append(np.count_nonzero(labels == labl))

        #create binary array of only the largest component
        binary_mask = np.zeros(labels.shape)
        binary_mask = np.where(labels == vol_list.index(max(vol_list))+1, 1, 0)

        #remove the largest component
        m = fill - binary_mask
        
        im_final[plane,...] = m
        
    return im_final

def correct(mask_alg,raw_img):
    '''
    Parameters
    ----------
    mask_alg : image
        mask image of all the algae.
    raw_img : image
        the contrast stretched image.

    Returns
    -------
    new_image : image
        the corrected image: contrast-stretched image without the algae
    
    Workflow
    --------
    
    for each plane take find the algae that not below the organism by comparing the masks 
    replace the value of the algae by the mean pixel value of the image (increase contrast)

    '''
    
    #initialize the corrected raw image
    
    new_image = np.zeros_like(raw_img) 
    
    for plane in range(np.shape(raw_img)[0]):

        #correct the image based on the substraction

        new_image[plane,...] = raw_img[plane,...] * (mask_alg[plane,...] == 0.)
        
        new_image[plane,...][new_image[plane,...] == 0] = np.mean(raw_img[plane,...]) 

    return new_image

def preprocessing(path):
    '''
    Function to preprocess the image

    Parameters
    ----------
    path : string
        path to find the image to preprocess.

    Returns
    -------
    c : image
        the preprocessed image.
    al : image
        the algae mask.
        
    Workflow
    --------
    - Load the image
    - Stretch the contrast
    - segment the algae
    - correct the image (remove the algae)

    '''
    im = imread(path)
    im = img_as_ubyte(im)
    
    print('Increasing the contrast...')
    print('--------------------------------------------------------------')
    img = stretch(im)
    
    print('Done with the contrast!')
    print('--------------------------------------------------------------')
    print('Segmenting the algae...')
    print('--------------------------------------------------------------')
    al = alg(img)
    
    print('Done with the algae!')
    print('--------------------------------------------------------------')
    print('Triyng to correct the image... ')
    
    c = correct(al,img)
    
    print('--------------------------------------------------------------')
    print('Done with the correction!')
    print('--------------------------------------------------------------')
    
    return c,al 


# Drift computation 

def dist(df_t,df_t1,points):
    '''
    Function to compute te difference in coordinate between 2 points

    Parameters
    ----------
    df_t : dataframe
        position of te algae at frame t.
    df_t1 : dataframe
        position of te algae at time t plus 1.
    points : list
        the position in the dataframe where to compute the distance.

    Returns
    -------
    dx,dy: int 
        Te difference in x and y between the 2 points.

    '''
    one = df_t.iloc[points[0]]
    two = df_t1.iloc[points[1]]

    dx = (two['centroid-1']-one['centroid-1'])
    dy = (two['centroid-0']-one['centroid-0'])
    
    return dx,dy 

def eucldist(df_t,df_t1,points):
    '''
    Function to compute the eucledian distance
    
    Parameters
    ----------
    df_t : dataframe
        position of te algae at frame t.
    df_t1 : dataframe
        position of te algae at time t plus 1.
    points : list
        the position in the dataframe where to compute the distance.

    Returns
    -------
    dist : int
        the eucledian distance between the 2 points.

    '''
    one = df_t.iloc[points[0]]
    two = df_t1.iloc[points[1]]

    diff = (two['centroid-1']-one['centroid-1'])**2 + (two['centroid-0']-one['centroid-0'])**2
    dist = np.sqrt(diff)
    return dist 

def distance(df_t,points):
    '''
    Function to compute the distance
    
    Parameters
    ----------
    df_t : dataframe
        position of the points you want to compute the distance
    points : list
        the position in the dataframe where to compute the distance.

    Returns
    -------
    dx,dy : int
        the difference in x and y between the 2 points.

    '''
    one = df_t.iloc[points[0]]
    two = df_t.iloc[points[1]]

    dx = two['X']-one['X']
    dy = two['Y']-one['Y']
    
    return dx,dy


def drift(df_t,df_t1,method = 'linear'):
    '''
    Parameters
    ----------
    df_t : dataframe
        position of te algae at frame t.
    df_t1 : dataframe
        position of te algae at time t plus 1.
    method : string, optional
        method to compute te best assignment. The default is 'linear sum'.

    Returns
    -------
    distnace: list
        the distances between all the assigned point from frame t and t plus 1.

    Workflow:
    --------
    
    Function to compute the drift between 2 time points.
        - Create the distance matrix i.e. the "distance" between every point
        - perform linear sum assignment on this matrix
        - extract the best assignment and compute the 'optimal distance' between the assigned points 
        
    '''
    #Initialize the cost matrix
    m = []
    
    row = df_t.index.values
    col = df_t1.index.values
    
    for t in row:
        for t_1 in col:
            m.append(eucldist(df_t,df_t1,[t,t_1])) #note that here the order of the arguments in the dist function matters 
            
    #reshape the matrix (might be a more efficient way to create it)
    
    m = np.asarray(m).reshape((len(row),len(col)))
    
    if method.lower() == 'linear':
        
        # Perform linear sum assignment

        row_ind, col_ind = linear_sum_assignment(m)
        
    elif method.lower() == 'bipartite':
        
        biadjacency_matrix = csr_matrix(m)
        
        row_ind,col_ind = min_weight_full_bipartite_matching(biadjacency_matrix)
   
    else:
        return 'Please input a valid method'
    
    # Initialize the optimal distance list
    
    distance = []
    
    # Besed on the result of the linear assignment compute it
    
    for i,j in zip(row_ind,col_ind):
        distance.append(dist(df_t,df_t1,[row[i],col[j]]))
        
    return distance

def global_drift(im,method = 'linear'):
    '''
    Parameters
    ----------
    im : image
        the image where there is drift.
    method : string, optional
        the method to compute the assignment. The default is 'linear sum'.

    Returns
    -------
    dx : list
        list of all the median displacement in x between each frames.
    dy : list
        list of all the median displacement in y between each frames.
        
    Function to compute the drift for a whole image plane per plane
    Workflow:
        - Loop over planes
        - compute the labels for the 2 images
        - call the drift computation function to extract the mean drift between plane t and t+1
        - append the mean drift of all points to a list
    Inputs:
        - the mask_image of all the algae for each time point
        - the method to map the points between 2 time points
    '''

    drifx = []
    drify = []
    
    for plane in range(np.shape(im)[0]-1):
        
        print(f'{(plane/np.shape(im)[0])*100:.2f} % done ...')
        
        lab = label(im[plane,...])
        props = regionprops_table(label_image=lab, properties=('centroid','area'))
        df_t = pd.DataFrame(props)
        
        # Filter the results to have plausible algae 
        # Here I am assuming that a "real algae" needs to have an area moe than 10 pixels² and less than 300
        
        df_t = df_t[df_t.area < 300].reset_index() #reset index to avoid having gaps in index value 
                                                #(it is used in the dist function)
        df_t = df_t[df_t.area > 10].reset_index()
        
        lab = label(im[plane+1,...])
        props = regionprops_table(label_image=lab, properties=('centroid','area'))
        df_t1 = pd.DataFrame(props) 
        df_t1 = df_t1[df_t1.area < 300].reset_index()
        df_t1 = df_t1[df_t1.area > 10].reset_index()
        
        e = drift(df_t,df_t1,method)
        x = [x[0] for x in e]
        y = [x[1] for x in e]
        
        drifx.append(np.median(x))
        drify.append(np.median(y))
        
    return drifx,drify

def filtering_drift(dx,dy):
    '''
    Parameters
    ----------
    dx : list
        list of all the median displacement in x between each frames.
    dy : list
        list of all the median displacement in y between each frames.
    Returns
    -------
    df_drift : dataframe
        dataframe containing:
            - begining frame of drift
            - end frame of drift (begining plus 1)
            - the direction of the drift (0 or 1)
            - the coordinate that changed
            - the displacement in pixels
    res : list
        a list of te frames where there is drift (used later in the pipeline).

    Workflow:
    --------
    
    Find where there is drift:
        - the criteria is that over the last 4 frames the median displacement is >3 (absolute value)
        - I use the media to avoid getting biased by planes where the pipeline failed
    
    - Create a list with 3 zeros
    - Loop over all values of drift in x and y
    - append to the 0 list the absolute value of the drift
    - If the median of the n-4 n frame is > 3 pixels than there is drift
        - in that case I look at the mean of the distance in those planes
        - The sign of this will give me the direction
    - I append to another list:
        -the start plane of the drift
        -the end plane of the drift
        - the direction
        - if it's in x or y
        - the median of the displacement
    - From this direc variable I have a begining and an end of drift but I
    need the intermediate value of drift. For that I loop over the interval of drift:
        - while the start is different from the end append to a list:
            - frame start
            - frame start 1  
            - the direction
            - the coordinate that changed
            - the value of the drift between these frames
    - The issue is that I will have multiple time the same value in the list created above
    to avoid this I filter the list for repeating values.
    '''
    
    lx = [0,0,0]
    ly = [0,0,0]
    
    
    counterx = 4
    countery = 4
    direc = []
    
    # loop over planes
    
    for i in dx:
        lx.append(np.abs(i))
        ly.append(np.abs(dy[countery-4]))
        
        if np.median(lx[counterx-4:counterx]) > 3 :
            
            if np.mean(dx[counterx-4:counterx]) < 0:
                direc.append([counterx-4,counterx,1,'x',np.median(lx[counterx-4:counterx])])
            else: 
                direc.append([counterx-4,counterx,0,'x',np.median(lx[counterx-4:counterx])])
    
        if np.median(ly[countery-4:countery]) > 3:
            
            if np.mean(dy[countery-4:countery]) < 0:
                direc.append([countery-4,countery,1,'y',np.median(ly[countery-4:countery])])
            else:
                direc.append([countery-4,countery,0,'y',np.median(ly[countery-4:countery])])
                               
        counterx += 1 
        countery += 1
        
    pos = []
    for value in direc:
        start = value[0]
        end = value[1]
    
        while start != end:
            if value[3] == 'y':
                pos.append([start,start+1,value[2],value[3],round(dy[start])])
            else:
                pos.append([start,start+1,value[2],value[3],round(dx[start])])
            start += 1
        
    res = []
    single_drift = []
    for j,i in enumerate(pos):
        if [i[0],i[1]] not in res:
            single_drift.append(i)
            res.append([i[0],i[1]])
            
    
    df_drift = pd.DataFrame(single_drift,columns=(['start','end','direction','coord','displacement']))
    
    return df_drift,res

## Visualizing the drift

def create_canva(df_drift,im):
    '''
    Parameters
    ----------
    df_drift : dataframe
        the dataframe that contaiins the information about the drift.
    im : image
        the processed image.

    Returns
    -------
    canva : the rescaled image
    diffx : int
        the difference in x to place the image.
    diffy : int
        the difference in y to place the image..
        
    Workflow:
        - Compute the evolution of 4 coordinate x0,y0 and xmax,ymaxx
        which are the extremities of the image.
        - Make them evolve according to the drift
        - find the maximal size of the frame 
        - create a canve accordingly
    '''
    orix = [0]
    oriy = [0]
    
    maxx = [np.shape(im)[1]-1]
    maxy = [np.shape(im)[2]-1]
    
    counterx = 1
    countery = 1
    
    for i in df_drift.iloc:
        if i.coord == 'x':
            if i.direction == 0:
                orix.append(orix[counterx-1] - np.abs(i.displacement))
                maxx.append(maxx[counterx-1] - np.abs(i.displacement))
            else:
                orix.append(orix[counterx-1] + np.abs(i.displacement))
                maxx.append(maxx[counterx-1] + np.abs(i.displacement))
                
            counterx +=1
        else:
            if i.direction == 0:
                oriy.append(oriy[countery-1] - np.abs(i.displacement))
                maxy.append(maxy[countery-1] - np.abs(i.displacement))
            else:
                oriy.append(oriy[countery-1] + np.abs(i.displacement))
                maxy.append(maxy[countery-1] + np.abs(i.displacement))
            countery += 1  
            
    lengx = np.abs(max(maxx)+1) + np.abs(min(orix)) + 1

    lengy = np.abs(max(maxy)) + np.abs(min(oriy)) + 1 
    
    diffx = lengx - np.shape(im[0])[0] 
    diffy = lengy - np.shape(im[0])[1] 
    
    canva = np.zeros((np.shape(im)[0],lengy,lengx))
    
    return canva,diffx,diffy

def place_img(canva,diffx,diffy,im,df_drift,res):
    
    '''
    Parameters
    ----------
    canva : image
        the canve where to 'paint' the image.
    diffx : int 
        the variable that tells where to place the first image in x.
    diffy : int
        the variable that tells where to place the first image in y.
    im : image
        the image to place in the canva.
    df_drift : dataframe
        the dataframe containing the value,direction and coordinate of the drift.
    res : list
        a list containing all the planes in which there is drift.

    Returns
    -------
    canva : image
        the rescaled image.

    Workflow:
    --------
    
    - Initialize the first frame (still to do: need a way to know how to place 
                                  the first image)
    - loop over planes
    - find which coordinate changed and in which direction it went
    - adjust the coordinates accordingly
    - return the 'painted' canva  
    '''
    
    # Initialize the first frame:

    ymin = 0
    ymax = np.shape(im[0,...])[0]
    
    xmin = diffx
    xmax = np.shape(canva[0,...])[1]
    
    canva[0,ymin:ymax,xmin:xmax] = im[0]
    counter = 0
    
    for plane in range(np.shape(im)[0]):
        print(f'Placing plane {plane} out of {np.shape(im)[0]} on the canva')
        
        if [plane,plane+1] in res:
            
            i = df_drift.iloc[counter]
            counter +=1
            if i.coord == 'x':
                if i.direction == 0:
                    xmin = xmin - np.abs(i.displacement)
                    xmax = xmax - np.abs(i.displacement) 
    
                    ymin = ymin 
                    ymax = ymax 
                    
                    canva[plane,ymin:ymax,xmin:xmax] = im[i.start] 
                    
                else:
                    xmin = xmin + np.abs(i.displacement)
                    xmax = xmax + np.abs(i.displacement)
    
                    ymin = ymin
                    ymax = ymax
    
                    canva[plane,ymin:ymax,xmin:xmax] = im[i.start]
            else:
                if i.direction == 0:
                    xmin = xmin 
                    xmax = xmax 
    
                    ymin = ymin - np.abs(i.displacement)
                    ymax = ymax - np.abs(i.displacement)
                    
                    canva[plane,ymin:ymax,xmin:xmax] = im[i.start]
                else:
    
                    xmin = xmin 
                    xmax = xmax 
    
                    ymin = ymin + np.abs(i.displacement)
                    ymax = ymax + np.abs(i.displacement)
                    canva[plane,ymin:ymax,xmin:xmax] = im[i.start]
        else:
            canva[plane,ymin:ymax,xmin:xmax] = im[plane]
    
    
    return canva