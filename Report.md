# Abstract 1000 characters 

*Trichoplax Adherens* is a marine organism from the Placozoa phylum. *Trichoplax* is composed of a bilayer of epithelium cells connected by fibrous cells. They self assemble and reproduce mainly by binary fission. This is explained by the cellular composition of the organism and the lack of organoic structures. The basal epithelium layer displays cilliated cells, which is thought to be the source of placozoan movement. 

*à completer*


# Core 18000 characters (including figure legends), 7 figures max

## Introduction

Introduced in 1883 by the German zoologist Franz Eilhard Schulze (ref ?), *Trichoplax adherens*, a flat crawling marine animal, consitutes one of the three member of the Placozoan phylum. The names come from the greek words θρίξ, *trich* = hair and  πλάξ, *plax* = plate. Indeed, the basal layer of epithelia is composed of cilliated 'hairy' cells. These cells are thought to be the cause of the movement of the organism (ref).  Being a multicellular self-assembled organism, *Trichoplax* feeds on microparticles present in their environment by absorbtion.

Although self organized, *Trichoplax* display the property of wound healing and seem to present collective behaviour (ref). This fascinating property might the basis of cellular communication ad tissue organisation found in higher order organism in the animalia kingdom. Furthermore, the displacement of a collective ensemble of cells poses exciting theoretical questions. Recently, authors showed a thermotaxis behaviour of *Trichoplax* without the evidence of a neural system (ref).
Experimentally, probing these properties poses some technical challenges. 
- Firstly, althought rich in gene number, *Trichoplax* lack stable immunofluorescent markers. As a consequence, most highly resolutive imaging approaches are not useful (find better word).
- Moreover, imaging a moving object present a technical challenge, mainly because the organism needs to stay in the field of view of the microscope. This leads to the need for a large field of view to limit camera movement and thus impact the resolution.

Combining these technical difficulties, experiements are usually carried out using bright field microscope recording temporal data. These experiments yield thus temporal data that require segmentation and tracking of a moving object in time. Segmentation is a classification problem where pixels in an image need to be categorized. From this classification one can compute properties of the connected regions, i.e. segmented objects. In the casse of tracking, we need to have information of the position in space and time of the object(s) in order to build trajectory. Such process cannot be performed by hand by the experimenter due to the large amount of data generated. Thus, the need for an automatic pipeline is present. 
However, segmentation and tracking are still not trivial to compute for a machine, in this report we explore various techniques and present limitation and error of different approaches. In the end we show how we developped **Trichomatic*** a python based-pipeline for the segmentation, tracking and quantification of plocozoa movement. We describe the different modules and present preliminary results for it's use. An illustration of the complete pipeline can be found in:

![[Placozoa tracking final.png]]

## Methods

Describe the different submodules: 

### Preprocessing

The preprocessing module has the purpose of maximizing the results of the segmentation. It is composed of four different operation on the image. The image is loaded, an initial segmentation is performed. The connected components are ranked by size and the largest component is used to create the preliminary organism mask. The other components are labeled as the algae mask. The two masks are used to correct the image. The algae segmented that are not present on the organism mask are removed from the image and replaced by the average background intensity measure of the image, allowing a global smoothing of the image. Finally, a contrast stretching is implemented. 

### Drift computation/correction

We define drift as ponctual movements of the camera. The drift is computed using static objects in the field of view: algae. We compute the drift as the distance between an object at frame t and the same object at frame t+1. The computation is separated in three different steps: algae identification, assignment and filtering of the results. We used the algae mask computed during the preprocessing in order to identify the algae. The objects are labeled and centroids are computed. In order to perform assignments, a distance matrix is established. A distance matrix computes a metric between each point in an image at frame t and every point in the next time point. The metric chosen for the pipeline is the eucledian distance in 2D, as described by equation 1. A threshold has been set to avoid missalignements: computed distances above a certain value are attributed an extreme value to guide the assignement. The threshold value is set as a parameter and is estimated from the movement of the camera during an experiment. The resulting matrix is used for the assignment. Two different methods have been implemented to compute the assignement: linear sum assignment (ref) and min weight bipartite graph (ref).

$$ d = \sqrt{(x_{(t+1)} - x_{(t)})^2+(y_{(t+1)} - y_{(t)})^2} \tag{eq1}$$

The assignment produces a list of optimal matches that is then used for determining drift. We compute a distance between the matched points for the two different coordinate using equations 2 and 3:

$$dx = x_{t+1} - x_{t} \tag{eq2}$$
$$dy = y_{t+1}-y_{t} \tag{eq3}$$

We define the drift as being a movement of at least five pixels for more than three consecutive frames. This threshold was set using experimental data and can be adjusted by the user as a parameter. In order to filter the data, a sliding window is perfromed in order to gather the frames that match the criteria. A dataframe is created containing information on the displacement vector: the direction of the displacement, the coordinate that was displaced and the amplitude of the movement. This dataframe are then used to correct the centroids of the corresponding frames. 

### Segmentation/ quantification

The segmentation was performed using intensity-based methods. Two different methods have been implemented: Otsu algorithm for intensity thresholding (ref) and Chan-Vese algorithm (ref).  They consist of blablabla.
Feature extraction is preferably performed using Chan-Vese algorithm for better performances and presegmentation during preprocessing was performed using the Otsu algorithm for faster computation.

### Postprocessing

The post-processing module consist of several filtering step in order to smooth the results of the segmentation. Moreover the post-processing module contains a correcting module for the camera movement. The centroid of the segmented objects are corrected for the planes where drift was detected according to the drift dataframe. A sliding window on the area of the objects is implemented to remove outliers and segmentation errors. The value superior or inferior to a certain trehsold to the mean of the sliding window are averaged. Finally, the wound properties are interpolated from the data using a UnivariateSpline algorithm from scippy (ref). 

### Imaging ***Trichoplax Adherens***
microscopy specs

## Results

### Measurement of the pipeline improvement 

#### How to measure improvement?

Measuring improvement when looking at data is a subjective process. How does one define improvement? Several metrics can be used: visually more relevant, smoother data, more robust to new data,... The intent of the pipeline is to provide a standardized and scalable analysis framework. In the context of the development of the pipeline an improvement metric was thus needed to be implemented. Moreover, improvement higly depends on the data considered and prior knowledge about it. In our case the pipeline was designed to be used on bright field images of ***Trichoplax Adherens*** moving in solution through time. In this case, improvement can be defined as results that further agree with litterature. 
In a more pragmatic view we decided to consider and measure improvement as a combination of several factors:
	- Firstly, as we consider a moving obnject at relatively short timescale improvement was defined as the presence or abscence of spikes in the data. The organsim wasn't subject to violent changes throughout the experiment - except wounding but in that case it is far few time frames and once - thus the data is expected to be smooth. 
	- Secondly, as a complete data-oriented metric we define imrpovement as the number of frames in which features can be seen. Indeed, one of the main goal of the pipeline is to analyse wound healing in ***Trichoplax***. For this, the wound needs to be a feature of the object . However, depending on the quality of the data this last operation can be more or less frequently possible. We thus decided to use this property as a metric to quantify the improvement of the pipeline.

We thus have semi-quantitave metrics in order to define the improvement of the pipeline on the data.

- How do you define improvement:
	- More visual
	- See more details
	- smoother data
	- definition depends on the data you have
- How for the data I have I defined imrpovement:
	- looked at smoothness of the outputs
	- abscence of spikes
	- time you can see the wound

#### The pipeline matches and improves manual adjustments of the image

The following question regarding imrpovement regards how to actually measure it. We defnied the different metrics used but we need to compare it to a standard value in order to extract a real improvement *per se*.
We constructed an image for such puprose, this image was manually adjusted to fit a "perfect scenario case". The contrast was manually adjusted for each frame in order to increase details and increase the resulting performance of the segmentation. This image will be thus use to benchmark the pipeline as the measure to match. Following what was previously stated, we computed different properties follwing the segmmentation of the image and compare the results. The result can be seen in Fig 1. 

- Compared the pipeline with manual contrasted image
- maybe add no operation at all on image
- we see that we can observe the wound for longer 
- the wholes (missing frames) are less and more spread

![[fig1.png]]
Figure 1: Improvement of the pipeline


We observe that the pipeline can match the results of the manually contrasted image and allow for a complete analysis. 

### Accounting and correcting for the movement of the camera

Following a moving organism means actualizing it's position for each time point acquired. However, following a moving organism in a moving environment means separating the two kind of movement to account only for the organims displacement. When acquiring a microscopy image we place the image on the microscope's objective referential. More precisely, the detecor referential i.e the number of pixels in x and in y where the image is recorded. However, the organism displaces itself on another referentiel: the media where it is. Indeed, the two referential can be the same, but only if the camera "sees" all the displacement of the organism. This last statement implies a large field of view, which in the case of bright field influences a lot the resolution of the image (bigger field of fiew, less resolution). In the experimental setup used to acquire the data, we decided to maximize the resolution as much as possible to be able to extract features. In this case, the organism often moved out of the field of view. Which implied that we had to move the camera and thus the field of view to follow it. In this configuration, our statement of the two referential being identical becomes false. When the camera moves a new referential is set which is different from the media referential. In order to correct for this, we designed a module to detect these camera movement on the image and correct for them. 

- How do you set the criteria for drift? 
- How do you correct for it?
- How do you verify it? 
- Introduce the concept of manual drift computation as ground truth


![[fig2.png]]
Figure 2: Correcting for the drift in the movie

the pipeline performed not as accuratly as the ground truth but still good enough

### Error measurement and limitations of the pipeline

How do you trust a result coming from an automatic pipeline? This question is at the center of the development of an analysis pipeline. One could look at the quality of the output and judge based on experience the reliability of the pipeline. However this approach is highly biased and not very robust. In this section we detail another approach we took concerning error estimation and current limitations of the pipeline. The question then becomes: how do you define an error? How do you compute it?
With the democratization of highly sophisticated and automatic tools such as machine learning algorithm this question was asked. These algorithm usually rely on very complex architectures and can be considered as black boxes that produce results. In order to estimate the performance and thus the rror of such approaches one can build a reference, also names ground-truth and compare the result of the algorithm against it. This ground-truth is usually build using experience on the data and used to estimate an error of the algorithm.
In our case, we want to estimate an error of performance of the pipeline. This error can have multipple meaning. We could look at the error on the segmentation, in this case the ground truth would be a hand made mask of the organism and the wound over time.
In this section we describe how we estimated the error on the drift detection and correction. We focus on this part as it influences multiple measurement downstream in the pipeline. Considering this error, we define the ground truth as the posiiton of the algae a different time measured by hand on the image. These coordinates were then used to compute an error and used to estimate the performance of the pipeline.

- How do you estimate the error (going back to ground truth)
- What is the error we are refereing to 
- Why is it necessary to estimate an error
- what does it mean that it's wrong ?
- Where does the error comes from ?
- Whant can be used to correct it
- still the presence of high noise in the data, not very robust
- high cumulative error although a mean/median error not so high
- presence of outlier: planes where there is a huge error 

![[fig3.png]]
Figure 3: Estimating the rror of the pipeline

The pipeline is still not very robust to new data. Moreover, even if the average error on the drift measurement is low the cumulative error on a whole movie represents still a lot of error (around 25% of the total image size). 

### Toward quantification of the movement of ***Trichoplax Adherens***

We saw during this report the general performances of the analysis pipeline and the current limitations around it. In this section we discuss the possibilities of quantification that it offers and how it can be used to uderstand the process of wound healing in ***Trichoplax Adherens***. 

- What can we do with the data? 
- What is the kind of question we are interested in ?
- What kind of plots can we do?
- Is the pipeline modulable ?
- The trajectory could give info on behaviour
- linking trajectory to other metrics could give more info on behaviour
- the area of the wound could reflect on organism physiology 
- carefull that the trend observed in area is biased by the 2D (folding)

![[fig4.png]]
Figure 4: Toward possible quantification of trichoplax
## Discussion

- Conclude
- what it will be used for
- where does limitations come from:
	- For the error:
		- assignment
		- detection of alagae
		- segmentation
	- Segmentation:
		- edge case
		- lack of contrast
		- algae that are touching the organism
	- Plotting:
		- smoothness of the data
		- bias of the 2D (folding of the organism)
- what needs to be optimized:
	- computation time
	- semgentation parameters
- what can be improved/ added
- incorporate in a microscope ? 

## Code availability

The complete pipeline is implemented inside the placozoa package developped during the 2022 Centuri Hackathon. The code used to produce the figures can be found on [Github](https://github.com/GuignardLab/placozoa_tracking). The package is under an open_source license BY-NC. The data used for the analysis and the development of the code are available on demand to the authors. 

## Aknowledgements

The author  would like to thank the two different teams that hosted me during the internship. The team Le Bivic at the Institut de biologie du developpmeent de Marseille (IBDM) and the Guignard Lab at the laboratoire d'informatique et systèmes (LIS). I would like to thank my supervisors Andrea Pasini and Léo Guignard for their guidance,discussions and feedback during the project and for their help writing this report. I would like to thank Philippe Roudot from the Institut de Mathématiques de Marseille (I2M) for the name of the pipeline Trichotomatic. 
# Annexes 5 pages maximum

- Description schématique des algos
- distribution des missing planes



