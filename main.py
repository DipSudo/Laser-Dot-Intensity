import cv2  
import numpy as np
import matplotlib.pyplot as plt

# Importing the left and right circular polarised images
LCP = cv2.imread("dots.png", cv2.IMREAD_GRAYSCALE)   #left circular polarised image
RCP = cv2.imread("dots.png", cv2.IMREAD_GRAYSCALE)   #right circular polarised image


# Convert to float
LCP = LCP.astype(np.float32)
RCP = RCP.astype(np.float32)



def background_normalise(image):
    
    """
    
    Normalise the image by dividing by a blurred version to remove background variations
    
    args:
        image: 2D numpy array of the image to normalise
        returns: 2D numpy array of the normalised image
        
    """

    # Large Gaussian blur to estimate background
    background = cv2.GaussianBlur(image, (101,101), 0)

    # Avoid divide-by-zero
    background[background == 0] = 1

    # Normalise
    normalised = image / background

    return normalised

# Normalising the left and right circular polarised images 
norm_LCP = background_normalise(LCP)
norm_RCP = background_normalise(RCP)

# Plotting the original and normalised images 
plt.figure(figsize=(12,4))

plt.subplot(131)
plt.title("Original LCP")
plt.imshow(LCP, cmap='gray')
plt.axis("off")

plt.subplot(132)
plt.title("Normalised LCP")
plt.imshow(norm_LCP, cmap='gray')
plt.axis("off")

plt.subplot(133)
plt.title("Normalised RCP")
plt.imshow(norm_RCP, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()