## Non-parametric fitting with gaussian process regression
## Matthew Streeter 2020

import numpy as np
from scipy.ndimage.filters import median_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
from scipy.interpolate import RectBivariateSpline

from scipy.ndimage import gaussian_filter


class GP_beam_fitter():
    ''' Object for storing image mask and contains methods for fitting a supplied image with a GP model
        Inputs: beam_mask(array) has a nonzero value where the image (supplied later) can be sampled to build the model
        N_samples(int) number of samples to take - more samples is more accurate but slower
        N_pred((Ny(int),Nx(int))) the downsampled image size to generate the fitted beam. 
        Asks the GP model to predict Nx*Ny points before interpolating onto input grid
    '''
    def __init__(self,beam_mask,N_samples = 1000, N_pred =(100,100)):
        self.beam_mask = beam_mask
        self.mask_ind = np.nonzero(beam_mask.flatten())
        self.N_samples = N_samples
        self.N_pred = N_pred
        
        # fitting kernel 
        kernel = 1**2* Matern(
            length_scale=0.1, length_scale_bounds=(1e-2, 10.0),nu=1.5
        ) + WhiteKernel()
        
        self.gp = GaussianProcessRegressor(kernel=kernel)
        self.x,self.y,self.XY = self._make_image_grid(beam_mask)
        self.x_pred,self.y_pred,self.XY_pred = self._make_image_grid(np.ones(N_pred))
        self._determine_pixel_weights()
        
        
    def _make_image_grid(self,img):
        ''' Makes a standard regular grid for an image'''
        x = np.linspace(-1,1,num=img.shape[1],endpoint=True)
        y = np.linspace(-1,1,num=img.shape[0],endpoint=True)
        [X,Y] = np.meshgrid(x,y)
        XY = np.array([X.flatten(),Y.flatten()]).T
        return x,y,XY
        
    def _determine_pixel_weights(self):
        ''' Provides weights for each pixel depending on how many  potential sample pixels are nearby
            useful for randomly selecting pixels but prefering ones where the data is sparse.
            '''
        Ny,Nx = np.shape(self.beam_mask)
        bmf = gaussian_filter(self.beam_mask.astype(float),(Ny/10,Nx/10))        
        pixel_w = 1/bmf.flatten()[self.mask_ind]
        self.pixel_w = pixel_w/np.sum(pixel_w)
        
    def fit_beam(self,image,median_kernel = 5):
        ''' Fits the beam using the stored mask and the data from the provided image
            input:
                image(2D array with same size as beam_mask) the image to fit
                median_kernel(int) =5 : kernel to median filter input image. None to not median filter

            outputs:
                beam_image 2D array fitted image using samples from mask region
                beam_unc 2D array: standard deviation error estimate from model
        '''

        # image is normalized to standard 0-1 range
        imgMax = np.max(image)
        imgMin = np.min(image)
        if median_kernel is not None:
            I  = (median_filter(image.astype(float),median_kernel).flatten()-imgMin)/(imgMax-imgMin)
        else:
            I  = (image.astype(float).flatten()-imgMin)/(imgMax-imgMin)

        # randonly select pixels with weighted choice
        I_index = np.arange(len(I))
        selected_index = np.random.choice(I_index[self.mask_ind],
                                          size=self.N_samples,replace=False,p=self.pixel_w)

        # train GP model with samples
        x_train = self.XY[selected_index,:]
        I_train = I[selected_index]
        self.gp.fit(x_train,I_train)
       
        # predict on reduced grid
        I_pred,I_pred_err = self.gp.predict(self.XY_pred,return_std=True)
        I_pred = I_pred.reshape(self.N_pred)
        I_pred_err = I_pred_err.reshape(self.N_pred)

        # interpolate onto output grid
        beam_image = RectBivariateSpline(self.y_pred,self.x_pred,I_pred)(self.y,self.x)*(imgMax-imgMin)+imgMin
        beam_unc = RectBivariateSpline(self.y_pred,self.x_pred,I_pred_err)(self.y,self.x)*(imgMax-imgMin)
        
        # estimate fit quality
        if median_kernel is not None:
            trans_image = median_filter(image,median_kernel)/beam_image
        else:
            trans_image = image/beam_image
        null_trans_vals = trans_image[np.nonzero(self.beam_mask)]
        null_trans_mean = np.mean(null_trans_vals)
        null_trans_rms = np.std(null_trans_vals,dtype=np.float64)
        print(f'Null transmission mean = {null_trans_mean:1.06f}')
        print(f'Null transmission rms = {null_trans_rms:1.06f}')

        return beam_image, beam_unc
        
