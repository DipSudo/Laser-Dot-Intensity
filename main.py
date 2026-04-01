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
        image: 2D array of the image to normalise
        returns: 2D array of the normalised image
        
    """

    # Large Gaussian blur to estimate background
    background = cv2.GaussianBlur(image, (101,101), 0)

    # Avoid divide-by-zero
    background[background == 0] = 1

    # Normalise
    normalised = image /background

    return normalised

# Normalising the left and right circular polarised images 
norm_LCP = background_normalise(LCP)
norm_RCP = background_normalise(RCP)

# # Plotting the original and normalised images 
# plt.figure(figsize=(12,4))

# plt.subplot(131)
# plt.title("Original LCP")
# plt.imshow(LCP, cmap='gray')
# plt.axis("off")

# plt.subplot(132)
# plt.title("Normalised LCP")
# plt.imshow(norm_LCP, cmap='gray')
# plt.axis("off")

# plt.subplot(133)
# plt.title("Normalised RCP")

# plt.imshow(norm_RCP, cmap='gray')
# plt.axis("off")

# plt.tight_layout()
# plt.show()

def detect_dots(image):
    
    """
    Detect bright circular dots in a normalised image using Hough Transform.
    
    Args:
        image: 2D array of normalised image 
    
    Returns:
        List of (x, y) coordinates of detected dots
    """
    # Convert to 8-bit
    img = (image / image.max() * 255).astype(np.uint8)
    
    # Smooth image to reduce noise
    blur = cv2.GaussianBlur(img, (9, 9), 1.5)
    
    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=10,
        param1=50,
        param2=15,
        minRadius=3,
        maxRadius=20
    )
    
    # Format output
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        dots = [(x, y) for x, y, r in circles]
    else:
        dots = []
    
    return dots


dots = detect_dots(norm_LCP)

print("Detected dots:", len(dots))

# Plotting detected dots on the normalised image
plt.figure(figsize=(6,6))
plt.imshow(norm_LCP, cmap='gray')

x = [d[0] for d in dots]
y = [d[1] for d in dots]

plt.scatter(x, y, color='red', s=50)
plt.title("Detected Dots")

plt.show()