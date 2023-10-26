import numpy as np
from scipy.signal import convolve2d

#---------------------------------
#converted matlab script from 
#Parthasarathy, R. Nat Methods 9, 724–726 (2012).
#---------------------------------

def radialcenter(I):
    # Extract dimensions of the input image
    Ny, Nx = I.shape

    # Create a grid of x-values representing each column's position relative to the image center
    xm_onerow = np.arange(-(Nx-1)/2.0 + 0.5, (Nx-1)/2.0 - 0.5 + 1)
    xm = np.tile(xm_onerow, (Ny-1, 1))

    # Create a grid of y-values representing each row's position relative to the image center
    ym_onecol = np.arange(-(Ny-1)/2.0 + 0.5, (Ny-1)/2.0 - 0.5 + 1)
    ym = np.tile(ym_onecol, (Nx-1, 1)).T

    # Calculate derivatives in the u and v directions
    dIdu = I[:-1, 1:] - I[1:, :-1]
    dIdv = I[:-1, :-1] - I[1:, 1:]

    # Define a 3x3 averaging filter
    h = np.ones((3, 3)) / 9.0
    
    # Smooth the derivatives using the averaging filter
    fdu = convolve2d(dIdu, h, mode='same')
    fdv = convolve2d(dIdv, h, mode='same')
    dImag2 = fdu**2 + fdv**2

    # Calculate m, avoiding division by zero
    denom = fdu - fdv
    m = np.where(denom != 0, -(fdv + fdu) / denom, 0)

    # Handle infinities and NaNs in m
    m[np.isnan(m)] = 0
    m[np.isinf(m)] = 10 * np.max(m[~np.isinf(m)])

    # Calculate the weighted centroid of the image gradients
    sdI2 = np.sum(dImag2)
    xcentroid = np.sum(dImag2 * xm) / sdI2
    ycentroid = np.sum(dImag2 * ym) / sdI2
    
    # Calculate weights based on the gradient magnitude and distance to the centroid
    w = dImag2 / np.sqrt((xm - xcentroid)**2 + (ym - ycentroid)**2)

    # Use the least squares method to determine the radial center
    xc, yc = lsradialcenterfit(m, b=ym - m*xm, w=w)

    # Adjust xc and yc to match Python's 0-based indexing
    xc += (Nx - 1) / 2.0
    yc += (Ny - 1) / 2.0

    # Rough measure of the particle width (second moment of I - min(I))
    Isub = I - np.min(I)
    px, py = np.meshgrid(np.arange(Nx), np.arange(Ny))
    xoffset, yoffset = px - xc, py - yc
    r2 = xoffset**2 + yoffset**2
    sigma = np.sqrt(np.sum(Isub * r2) / np.sum(Isub)) / 2.0

    return xc, yc, sigma

def lsradialcenterfit(m, b, w):
    # Least squares solution to determine the radial symmetry center
    
    wm2p1 = w / (m**2 + 1)
    sw = np.sum(wm2p1)
    smmw = np.sum(m**2 * wm2p1)
    smw = np.sum(m * wm2p1)
    smbw = np.sum(m * b * wm2p1)
    sbw = np.sum(b * wm2p1)
    
    # Calculate the determinants for the least squares solution
    det = smw**2 - smmw * sw
    xc = (smbw * sw - smw * sbw) / det
    yc = (smbw * smw - smmw * sbw) / det
    
    return xc, yc

