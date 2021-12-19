import numpy as np

def algorithm1(meanColorNormalized, sigmaSquared):
    """
    input: - mean color normalized (in [0,1])
           - variance (sigma^2)
    
    output: - rgb triple of drawn color normalized (in [0,1])

    desc: implements algorithm1 from 
    https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Kim_Learning_Not_to_CVPR_2019_supplemental.pdf
    """
    C = []
    for cm in meanColorNormalized:
        while True:
            c = np.random.normal(cm, np.sqrt(sigmaSquared), 1)
            if 0 < c and c < 1:
                C.append(c)
                break             
    return (C[0],C[1],C[2])


def colorizeRandomly(greyscaleImage, variance, sharpEdges):
    """
    input: - a grey scale image of the shape (28,28)
           - variance (sigma^2)
           - a bool whether or not to sharp the edges (uniform color)
    output: - the unbiased colorized image
    """
    # we simply set the pixels to the maximum which are not zero
    # (other options would be to use some thresholds)
    if sharpEdges == True:
        for i in range(28):
            for j in range(28):
                if greyscaleImage[i][j] != 0:
                    greyscaleImage[i][j] = 255
    
    # draw a color uniformly from predefined colors
    meanColors = np.array([
        (220,20,60),
        (0,128,128),
        (253,233,16),
        (0,149,182),
        (237,145,33),
        (145,30,188),
        (70,240,240),
        (250,197,187),
        (210,245,60),
        (128,0,0)
    ])
    numberOfRows = meanColors.shape[0]
    randIndex = np.random.choice(numberOfRows)
    sampledMeanColor = meanColors[randIndex,:]
    
    # apply jitter to drawn color
    sampledMeanColor = algorithm1((sampledMeanColor[0]/255,
                                   sampledMeanColor[1]/255,
                                   sampledMeanColor[2]/255)
                                  ,variance)
    
    # compute effective color
    rLayer = np.rint(np.array((greyscaleImage/255)*(sampledMeanColor[0]*255)))
    gLayer = np.rint(np.array((greyscaleImage/255)*(sampledMeanColor[1]*255)))
    bLayer = np.rint(np.array((greyscaleImage/255)*(sampledMeanColor[2]*255)))

    # compute colorized image
    colorizedImage = []
    for i in range(28):
        for j in range(28):
            colorizedImage.append([int(rLayer[i][j]),int(gLayer[i][j]),int(bLayer[i][j])])

    colorizedImage = np.array(colorizedImage)
    colorizedImage = np.reshape(colorizedImage, (28,28,3))  
    return colorizedImage

    

def generateUnbiasedTestSet(images, variance, sharpEdges):
    testSet = []
    for image in images:
        testSet.append(colorizeRandomly(image, variance, sharpEdges))
    return testSet