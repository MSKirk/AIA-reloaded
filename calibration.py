import numpy as np
import cv2
from astropy.io import fits
import os


class AIAPrep:
    # Promoting to the level 1.5 standard
    # 1) Plate scale, 2)rotation & 3)shift
    # 4) Set floor to zero, Rescale to 32 bit float
    # See http://jsoc.stanford.edu/doc/data/aia_test/SDOD0045_v10_AIA_plan_for_producing_and_distri_AIADataP_gPlanv1-6.pdf

    def __init__(self, filename, level=1.5, cropsize=None, rice=True):
        # The aia image size is fixed by the size of the detector. For AIA raw data, this has no reason to change.
        self.aia_image_size = 4096
        self.cropsize = cropsize
        self.input_file = filename
        self.data, self.header = aia_fits_read(self.input_file, rice=rice)

        self.aiaprep()
        self.aia_lev15_header_update()

        if cropsize:
            self.crop_image()

    def aiaprep(self):
        # Divide by exposure time, avoid NaNs
        if self.header['EXPTIME'] <= 0:
            self.data /= np.inf
        else:
            self.data /= self.header['EXPTIME']

        # Target scale is 0.6 arcsec/px
        target_scale = 0.6
        scale_factor = self.header['CDELT1'] / target_scale
        # Center of rotation at reference pixel converted to a coordinate origin at 0
        reference_pixel = [self.header['CRPIX1'] - 1, self.header['CRPIX2'] - 1]
        # Rotation angle with openCV uses coordinate origin at top-left corner. For solar images in numpy we need to invert the angle.
        angle = -self.header['CROTA2']
        # Run scaled rotation. The output will be a rotated, rescaled, padded array.
        if all(ii > 0 for ii in reference_pixel+[scale_factor]):
            prepdata = scale_rotate(self.data, angle=angle, scale_factor=scale_factor, reference_pixel=reference_pixel)
        else:
            prepdata = self.data

        prepdata[prepdata < 0] = 0

        self.data = np.float32(prepdata)

    def aia_lev15_header_update(self):
        # update the header for level 1.5 corrections

        #plate scale
        self.header['CDELT1'] = 0.6
        self.header['CDELT2'] = 0.6

        #center pixel
        self.header['CRPIX1'] = 2048.5
        self.header['CRPIX2'] = 2048.5

        #Sun Size
        self.header['r_sun'] = self.header['rsun_obs'] / self.header['cdelt1']

        #Level Number
        self.header['lvl_num'] = 1.5

        #bits per pixel
        self.header['bitpix'] = -32

        #Define rotation matrix
        self.header['pc2_1'] = 0.0
        self.header['pc1_2'] = 0.0
        self.header['pc2_2'] = 1.0
        self.header['pc1_1'] = 1.0

        # Data Values
        self.header['DATAMIN'] = np.nanmin(self.data)
        self.header['DATAMAX'] = np.nanmax(self.data)
        self.header['DATAMEDN'] = np.nanmedian(self.data)
        self.header['DATAMEAN'] = np.nanmean(self.data)
        self.header['DATARMS'] = np.nanstd(self.data)

        # SDO documentation say these keys meant to be discarded:
        # http://jsoc.stanford.edu/~jsoc/keywords/AIA/AIA02840_K_AIA-SDO_FITS_Keyword_Document.pdf
        self.header.remove('OSCNMEAN')
        self.header.remove('OSCNRMS')

    def crop_image(self):
        center = ((np.array(self.data.shape) - 1) / 2.0).astype(int)
        half_size = int(self.cropsize / 2)
        self.data = self.data[center[1] - half_size:center[1] + half_size,
                   center[0] - half_size:center[0] + half_size]

        #new center pixel
        self.header['CRPIX1'] = self.data.shape[0]/2 + 0.5
        self.header['CRPIX2'] = self.data.shape[1]/2 + 0.5

    def write_prepped(self, save_path=os.path.curdir, save_name=None):

        if not save_name:
            save_name = os.path.basename(self.input_file).replace('lev1', 'lev15')

		self.header = nan_blaster(self.header)

        hdu = fits.CompImageHDU(self.data, self.header)        
        hdu.verify('silentfix')
        hdu.writeto(os.path.join(save_path,save_name), overwrite=True)
			
    # FUTURE --->> update the header for level 1.6 corrections
    #def aia_lev16_header_update(self):

    # Corrections to the level 1.6 standard
    # PSF and diffraction correction
    # Effective Area correction
    # Rescale to integer; saved as 32bit Int

    #def aia16prep(self):


def scale_rotate(image, angle=0, scale_factor=1, reference_pixel=None):
    """
    Perform scaled rotation with opencv. About 20 times faster than with Sunpy & scikit/skimage warp methods.
    The output is a padded image that holds the entire rescaled,rotated image, recentered around the reference pixel.
    Positive-angle rotation will go counterclockwise if the array is displayed with the origin on top (default),
    and clockwise with the origin at bottom.

    :param image: Numpy 2D array
    :param angle: rotation angle in degrees. Positive angle  will rotate counterclocwise if array origin on top-left
    :param scale_factor: ratio of the wavelength-dependent pixel scale over the target scale of 0.6 arcsec
    :param reference_pixel: tuple of (x, y) coordinate. Given as (x, y) = (col, row) and not (row, col).
    :return: padded scaled and rotated image
    """
    array_center = (np.array(image.shape)[::-1] - 1) / 2.0

    if reference_pixel is None:
        reference_pixel = array_center

    # convert angle to radian
    angler = angle * np.pi / 180
    # Get basic rotation matrix to calculate initial padding extent
    rmatrix = np.matrix([[np.cos(angler), -np.sin(angler)],
                         [np.sin(angler), np.cos(angler)]])

    extent = np.max(np.abs(np.vstack((image.shape * rmatrix,
                                      image.shape * rmatrix.T))), axis=0)

    # Calculate the needed padding or unpadding
    diff = np.asarray(np.ceil((extent - image.shape) / 2), dtype=int).ravel()
    diff2 = np.max(np.abs(reference_pixel - array_center)) + 1
    # Pad the image array
    pad_x = int(np.ceil(np.max((diff[1], 0)) + diff2))
    pad_y = int(np.ceil(np.max((diff[0], 0)) + diff2))

    padded_reference_pixel = reference_pixel + np.array([pad_x, pad_y])
    #padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=(0, 0))
    padded_image = aia_pad(image, pad_x, pad_y)
    padded_array_center = (np.array(padded_image.shape)[::-1] - 1) / 2.0

    # Get scaled rotation matrix accounting for padding
    rmatrix_cv = cv2.getRotationMatrix2D((padded_reference_pixel[0], padded_reference_pixel[1]), angle, scale_factor)
    # Adding extra shift to recenter:
    # move image so the reference pixel aligns with the center of the padded array
    shift = padded_array_center - padded_reference_pixel
    rmatrix_cv[0, 2] += shift[0]
    rmatrix_cv[1, 2] += shift[1]
    # Do the scaled rotation with opencv. ~20x faster than Sunpy's map.rotate()
    rotated_image = cv2.warpAffine(padded_image, rmatrix_cv, padded_image.shape, cv2.INTER_CUBIC)

    return rotated_image


# Alternate padding method. On AIA, it is ~6x faster than numpy.pad used in Sunpy's aiaprep
def aia_pad(image, pad_x, pad_y):
    newsize = [image.shape[0]+2*pad_y, image.shape[1]+2*pad_x]
    pimage = np.empty(newsize)
    pimage[0:pad_y,:] = 0
    pimage[:,0:pad_x]=0
    pimage[pad_y+image.shape[0]:, :] = 0
    pimage[:, pad_x+image.shape[1]:] = 0
    pimage[pad_y:image.shape[0]+pad_y, pad_x:image.shape[1]+pad_x] = image
    return pimage


# modularize the fits reading function    
def aia_fits_read(fitsfile, rice=True):

    hdul = fits.open(fitsfile)
    if rice:
        hdu = hdul[1]
    else:
        hdu = hdul[0]
    hdu.verify('silentfix')
    header = hdu.header
    data = hdu.data.astype(np.float64)

    return data, header

# Reads the header from the file and sanitizes it to Fits standards    
def nan_blaster(fitsheader, delete=True):

        # Remove newlines from comment and history
	if 'comment' in fitsheader:
		fitsheader['comment'] = fitsheader['comment'].replace("\n", "")
	if 'history' in fitsheader:
		fitsheader['history'] = fitsheader['history'].replace("\n", "")

	badkeys = []

	# finds header keywords that are NaN
	for key, value in fitsheader.items():
		if type(value) in (int, float):
			if np.isnan(value):
				badkeys += [key]
				
		if type(value) == str:
			if value.strip().lower() == 'nan':
				badkeys += [key]

	for key in badkeys:
		if delete:		
			fitsheader.pop(key, None)
		else:
			fitsheader.set(key, 0)

    return fitsheader


    


