import cv2  
import numpy as np
import matplotlib.pyplot as plt

# Importing the left and right circular polarised images
LCP = cv2.imread("dots.png", cv2.IMREAD_GRAYSCALE)   #left circular polarised image
# RCP = cv2.imread("dots.png", cv2.IMREAD_GRAYSCALE)   #right circular polarised image 

def simulate_RCP(LCP, variation=0.03, noise=0.01):
    
    """
    Simulate an RCP image from LCP for testing purposes
    
    """
    
    # Simulate polarisation contrast
    RCP = LCP * (1 - variation)
    
    # Add small random noise
    noise_map = np.random.normal(0, noise * np.std(LCP), LCP.shape)
    
    RCP = RCP + noise_map
    
    return RCP.astype(np.float32)

LCP = LCP.astype(np.float32)  # Convert to float for processing 
# RCP = RCP.astype(np.float32)
RCP = simulate_RCP(LCP)      # Simulate RCP from LCPfor testing purposes

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

# def detect_dots_Hough(image):       
    
#     """
#     Detect bright circular dots in a normalised image using Hough Transform (More effective than thresholding for circular features in this application)
    
#     """
#     # Convert to 8-bit
#     img = (image / image.max() * 255).astype(np.uint8)
    
#     # Smooth image to reduce noise
#     blur = cv2.GaussianBlur(img, (9, 9), 1.5)
    
#     # Detect circles using Hough Transform (Found to be more effective than blob detection for this application )
#     circles = cv2.HoughCircles(
#         blur,
#         cv2.HOUGH_GRADIENT,
#         dp=1.0,
#         minDist=10,
#         param1=50,
#         param2=15,
#         minRadius=3,
#         maxRadius=20
#     )
    
#     # Format output
#     if circles is not None:
#         circles = np.round(circles[0]).astype(int)
#         dots = [(x, y) for x, y, r in circles]
#     else:
#         dots = []
    
#     return dots


# dots = detect_dots_Hough(norm_LCP)

# print("Detected dots:", len(dots))

# # Plotting detected dots on the normalised image
# plt.figure(figsize=(6,6))
# plt.imshow(norm_LCP, cmap='gray')

# x = [d[0] for d in dots]
# y = [d[1] for d in dots]

# plt.scatter(x, y, color='red', s=50)
# plt.title("Detected Dots")

# plt.show()

def detect_dots_blob(image):
    
    """
    Detect bright circular blobs in a normalized image.
    Returns sub-pixel centers of detected blobs.
    
    This method was fouund to be more effective than Hough Transform (better than thresholding for circular features) for this application, as it can better handle varying blob sizes and intensities.
    """
    # Convert normalized image to 8-bit
    img_uint8 = (image / image.max() * 255).astype(np.uint8)
    
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255  # Detect bright blobs
    params.filterByArea = True
    params.minArea = 50      # Minimum blob size (This value got rid of small noise blobs)
    params.maxArea = 5000   # Maximum blob size
    params.filterByCircularity = True
    params.minCircularity = 0.7  
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    params.filterByConvexity = False  # Optional
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(img_uint8)
    
    # Extract (x, y) coordinates
    dots = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
    
    return dots, img_uint8

# Detect blobs
LCP_dots, LCP_img = detect_dots_blob(norm_LCP)
RCP_dots, RCP_img = detect_dots_blob(norm_RCP)

print("Detected LCP dots:", len(LCP_dots))
print("Detected RCP dots:", len(RCP_dots))

# # Plotting Blob Detection Results
# plt.figure(figsize=(6, 6))

# plt.subplot(131)
# plt.title("Original")
# plt.imshow(LCP, cmap='gray')

# plt.subplot(132)
# plt.imshow(LCP_img, cmap='gray')
# plt.title("LCP Blob Detection")
# for cx, cy in LCP_dots:
#     plt.plot(cx, cy, 'ro', markersize=4)

# plt.subplot(133)
# plt.title("RCP Blob Detection")
# plt.imshow(RCP_img, cmap='gray')
# for cx, cy in RCP_dots:
#     plt.plot(cx, cy, 'ro', markersize=4)

# plt.tight_layout()
# plt.show()

def extract_intensity(image, dots, radius=6):
    
    """
    Extract mean intensity around each detected dot
    
    args:
        image : normalised image
        dots  : list of (x,y) coordinates
        radius: region size around each dot
        
    returns:
        array of mean intensities
    """
    
    intensities = []

    for (x, y) in dots:
        
        # Create circular mask
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
        
        # Mean intensity inside dot
        mean_val = cv2.mean(image, mask=mask)[0]
        intensities.append(mean_val)

    return np.array(intensities)


# Extract intensities
intensity_LCP = extract_intensity(norm_LCP, LCP_dots)
intensity_RCP = extract_intensity(norm_RCP, RCP_dots)

print("Intensity extraction complete")

difference = intensity_LCP - intensity_RCP
