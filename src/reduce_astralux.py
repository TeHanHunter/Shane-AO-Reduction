# -*- coding: utf-8 -*-
import os
import numpy as np
from astropy.io import fits
import pickle
import glob
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
def get_sigma_mad(x):
    med = np.median(x)
    return 1.4826*np.median(np.abs(x-med))

from scipy.ndimage import median_filter, gaussian_filter
def guess_gaussian_parameters(d):
    """
    This image guesses the maximum intensity of the
    image by obtaining a smoothed version of the image
    via median + gaussian filtering. Then, finds the
    maximum of the smoothed image. Using the median-filtered
    image, the algorithm also estimates the width of the PSF and
    with this value and an estimate of the volume, the amplitude of
    such gaussian.

    Input

    d       Numpy array that contains the values of the pixels.

    Output

    x0      x coordinate of the image maximum

    y0      y coordinate of the image maximum

    sigma   Width of the PSF

    A       Estimated amplitude of the gaussian function
    """

    # First, smooth the image with a median filter. For this, find
    # the optimal window as the square-root of the geometric mean
    # between the sizes of the array. This step is good to kill outliers:
    window = int(np.sqrt(np.sqrt(d.shape[0]*d.shape[1])))
    if window % 2 == 0:
        window = window + 1
    d_mfiltered = median_filter(d,size = window)

    # Next, smooth it with a gaussian filter:
    d_gfiltered = gaussian_filter(d_mfiltered,sigma = window)

    # Now, find the maximum of this image:
    y0, x0 = np.where(d_gfiltered == np.max(d_gfiltered[140:450,150:450]))

    # Take the first element. This helps in two cases: (1) only one maximum has
    # been found, the outputs are numpy arrays and you want to extract the numbers
    # and (2) in case there are several maximums (this has not happened so
    # far but could in principle), pick the first of them:
    y0 = y0[0]
    x0 = x0[0]

    # Now estimate the width of the PSF by taking a "cross-plot" using the
    # maximum values found:
    x_cut = d[:,int(x0)]
    sigma_x = (np.sum(x_cut*(np.abs(np.arange(len(x_cut))-y0)))/np.sum(x_cut))/3.
    y_cut = d[int(y0),:]
    sigma_y = (np.sum(y_cut*(np.abs(np.arange(len(y_cut))-x0)))/np.sum(y_cut))/3.
    sigma = np.sqrt(sigma_x*sigma_y)

    # (Under) estimate amplitude assuming a gaussian function:
    A = (np.sum(d-np.median(d))/(2.*np.pi*sigma**2))

    return x0,y0,sigma,2.*A

def moffat(x,y,A,x0,y0,sigma,beta):
    first_term = ((x-x0)**2 + (y-y0)**2)/sigma**2
    return A*(1. + first_term)**(-beta)

def assymetric_gaussian(x, y, A, x0, y0, sigma_x, sigma_y, theta):
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return A*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) +\
                        c*((y-y0)**2)))

def gaussian(x, y, A, x0, y0, sigma):
    a = ((x-x0)**2 + (y-y0)**2)/(2.*(sigma**2))
    return A*np.exp(-a)

import lmfit

def modelPSF(params,mesh):
    W = params['W'].value
    ag = (1.-W)*assymetric_gaussian(mesh[0],mesh[1],params['Ag'].value,params['x0'].value,\
                        params['y0'].value,params['sigma_x'].value,params['sigma_y'].value,\
                        params['theta'].value)
    mof = W*moffat(mesh[0],mesh[1],params['Am'].value,params['x0'].value,\
                        params['y0'].value,params['sigma_m'].value,params['beta'].value)
    return ag+mof+params['bkg'].value

def fitPSF(d,x0,y0,sigma,A):
    """
    This function fits a sum of a rotated gaussian plus a
    moffat profile.
    """
    mesh = np.meshgrid(np.arange(d.shape[0]),np.arange(d.shape[1]))
    flattened_d = d.flatten()

    def residuals(params):
        return flattened_d - (modelPSF(params,mesh)).flatten()

    prms = lmfit.Parameters()
    prms.add('x0', value = x0, min = 0, max = d.shape[0], vary = True)
    prms.add('y0', value = y0, min = 0, max = d.shape[1], vary = True)
    prms.add('W' , value = 0.5, min = 0, max = 1., vary = True)
    prms.add('Ag', value = A, min = 0, max = np.sum(d-np.median(d)), vary = True)
    prms.add('Am', value = A, min = 0, max = np.sum(d-np.median(d)), vary = True)
    prms.add('sigma_x', value = sigma, min = 0, max = d.shape[0]/3., vary = True)
    prms.add('sigma_y', value = sigma, min = 0, max = d.shape[1]/3., vary = True)
    prms.add('sigma_m', value = sigma, min = 0, max = d.shape[1]/3., vary = True)
    prms.add('beta', value = 1., min = 0, max = 10.)
#    prms.add('bkg', value = np.median(d), min = np.median(d)-10*get_sigma_mad(d), \
#                    max = np.median(d)+10*get_sigma_mad(d), vary = True)
    prms.add('bkg', value = np.median(d), vary = True)
    prms.add('theta', value = np.pi/4., min = 0, max = np.pi)
    result = lmfit.minimize(residuals, prms)
    return result.params

#################### USER DEFINITIONS #######################

if __name__ == '__main__':
    data_folder = '/Users/tehan/Downloads/'
    # files = glob.glob(f'{data_folder}*.fits')
    files = glob.glob(f'{data_folder}TOI_5916_final.fits')
    for i in range(len(files)):
        filename = os.path.basename(files[i])
        # Minimum magnitude contrast to be explored:
        min_m = 0
        # Maximum magnitude contrast to be explored:
        max_m = 10
        # Contrast steps:
        contrast_steps = 100
        # Scale of the image in arcsecs/pixel:
        scale = 33*1e-3

        #############################################################
        print('\n\t     AstraLux contrast curve generator')
        print('\t-----------------------------------------------')
        print('\tAuthors: Nestor Espinoza (nespino@astro.puc.cl)')
        print('\t         Andres Jordan (ajordan@astro.puc.cl)\n')
        # Create output directory if non-existent for the current image:
        out_dir = data_folder + filename.split('.')[0] + '/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # If not already done, model the input image. If already done,
        # obtain saved data:
        # if not os.path.exists(out_dir+'/model_image.fits'):
        print('\t > Modelling the PSF...')
        # First, extract image data:
        d,h = fits.getdata(data_folder+filename, header=True)
        # print(np.shape(d))

        # Guess centroid by maximum intensity; also estimate approximate
        # width of the PSF by weighted median-absolute distance from the
        # estimated center and use it to estimate amplitude:
        x0,y0,sigma,A = guess_gaussian_parameters(d)
        # sigma = 30.0 #Use this if you're getting fwhm of 0
        # print(x0,y0,sigma,A)
        # Estimate model of moffat + rotated gaussian:
        out_params = fitPSF(d,x0,y0,sigma,A)
        # Save output parameters:
        #fout = open(out_dir+'/out_params.pkl','wb')
        #pickle.dump(out_params,fout)
        #fout.close()

        # Generate model image:
        model = modelPSF(out_params,\
                               np.meshgrid(np.arange(d.shape[0]),np.arange(d.shape[1])))
        # Generate residual image:
        res = model - d

        # Save images:
        print('\t > Saving results...')
        fits.PrimaryHDU(model).writeto(out_dir+'/model_image.fits', overwrite=True)
        fits.PrimaryHDU(d).writeto(out_dir+'/original_image.fits', overwrite=True)
        fits.PrimaryHDU(res).writeto(out_dir+'/residual_image.fits', overwrite=True)
        # else:
        #     print('\t > PSF already modelled. Extracting data...')
        #     # If everything already done, read data:
        #     model = fits.getdata(out_dir+'model_image.fits')
        #     d = fits.getdata(out_dir+'original_image.fits')
        #     res = fits.getdata(out_dir+'residual_image.fits')
        #     par = open(out_dir+'out_params.pkl','r')
        #     out_params = pickle.load(par)
        #     par.close()

        # Define the step in radius at which we will calculate the contrasts. This is
        # calculated in terms of the "effective FWHM", which we calculate numerically from
        # the model PSF, by trying different radii and angles and finding the positions at which
        # the flux is half the peak flux.
        max_flux_model = np.max(model)
        print(max_flux_model)
        #radii = np.linspace(0,50.,100) #alternate for trouble targets
        radii = np.linspace(0,5.*((out_params['sigma_x'].value+out_params['sigma_y'].value)/2.),100)
        print(radii)
        thetas = np.linspace(0,2*np.pi,100)
        fwhms = np.zeros(len(thetas))
        for j in range(len(thetas)):
            for i in range(len(radii)):
                c_x = out_params['x0'].value + radii[i]*np.cos(thetas[j])
                c_y = out_params['y0'].value + radii[i]*np.sin(thetas[j])
                if model[int(c_y),int(c_x)]<max_flux_model/2.:
                   fwhms[j] = np.sqrt((out_params['x0'].value-int(c_x))**2 + (out_params['y0'].value-int(c_y))**2)
                   break
        print('\t Effective FWHM:',np.median(fwhms),'+-',np.sqrt(np.var(fwhms)),' pix (',np.median(fwhms)*scale,'+-',np.sqrt(np.var(fwhms))*scale,' arcsecs)')
        radii_step = np.median(fwhms)
        N = np.median(fwhms)

        # Convert the radius step to int:
        radii_step = int(radii_step)
        print(radii_step)
        # Get centroids:
        x0,y0 = out_params['x0'].value,out_params['y0'].value

        # Remove estimated background from original image:
        d = d - out_params['bkg'].value

        # Now generate 5-sigma contrast curves. For this, first find
        # closest distance to edges of the image:
        right_dist = int(np.floor(np.abs(x0 - d.shape[0])))-N
        left_dist = int(np.ceil(x0))-N
        up_dist = int(np.floor(np.abs(y0 - d.shape[1])))-N
        down_dist = int(np.ceil(y0))-N
        #max_radius = np.min([right_dist,left_dist,up_dist,down_dist])
        max_radius = 213. #this is where quality starts degrading

        # Generate the contrast curve by, for a given radius and using the
        # residual image, injecting fake sources with the same parameters as
        # the fitted PSF but scaled in order to see at what scale (i.e., magnitude)
        # we detect the injected signal at 5-sigma). A detection is defined if
        # more than 5 pixels are above the 5-sigma noise level of this residual
        # image at that position.

        # First, define the radii that will be explored:
        radii = np.arange(radii_step,max_radius,radii_step)

        # Initialize arrays that will save the contrast curves:
        contrast = np.zeros(len(radii))
        contrast_err = np.zeros(len(radii))

        # Initialize magnitude contrasts to be explored:
        possible_contrasts = np.linspace(min_m,max_m,contrast_steps)

        # Now inject fake source on the images at each position, and see when we
        # recover it. First, set background of the model to zero:
        out_params['bkg'].value = 0.0

        print('\t > Generating contrast curves...')
        for i in range(len(radii)):
            # Define the number of angles that, given the radius, have
            # independant information:
            if radii[i] != 0:
                n_thetas = np.min([int((2.*np.pi)/((2.*N)/radii[i])),30])
                thetas = np.linspace(0,2*np.pi,n_thetas)
            # Generate vector that saves the threshold functions
            # at a given angle:
            c_contrast = np.zeros(len(thetas))
            for j in range (len(thetas)):
                # Get current pixel to use as center around which we will
                # extract the photometry:
                c_x = x0 + int(np.round(radii[i]*np.cos(thetas[j])))
                c_y = y0 + int(np.round(radii[i]*np.sin(thetas[j])))

                # Get nxn sub-image at the current pixel:
        #        c_subimg = res[c_y-(N/2)-1:c_y+(N/2),\
        #                       c_x-(N/2)-1:c_x+(N/2)]
                c_subimg = res[int(c_y-(N/2)-1):int(c_y+(N/2)),int(c_x-(N/2)-1):int(c_x+(N/2))]

                # Estimate the (empirical) standard-deviation of the pixels
                # in the box:
                sigma = np.sqrt(np.var(c_subimg))

                # Set the model PSF at the center of the current position:
                out_params['x0'].value = c_x
                out_params['y0'].value = c_y

                # Generate the fake source. We will scale it below to match
                # different contrasts:
                fake_signal = modelPSF(out_params,\
                              np.meshgrid(np.arange(d.shape[0]),np.arange(d.shape[1])))

                for k in range(len(possible_contrasts)):
                    # Generate the scaling factor:
                    scaling_factor = 10**(possible_contrasts[k]/2.51)
                    # Construct fake image on top of the residual image, cut the portion under
                    # analysis:
        #            fake_image = (res + (fake_signal/scaling_factor))[c_y-(N/2)-1:c_y+(N/2),\
        #                                                              c_x-(N/2)-1:c_x+(N/2)]
                    fake_image = (res + (fake_signal/scaling_factor))[int(c_y-(N/2)-1):int(c_y+(N/2)),\
                                                                      int(c_x-(N/2)-1):int(c_x+(N/2))]
                    # If our detection limit (i.e., 5 pixels or more are above 5-sigma) is not accomplished,
                    # then the source cannot be detected and this defines our 5-sigma contrast:
                    if (len(np.where(fake_image>5*sigma)[0])<5):
                        if k != 0:
                            c_contrast[j] = possible_contrasts[k-1]
                        else:
                            c_contrast[j] = 0.0
                        break


            idx = np.where((~np.isnan(c_contrast))&(c_contrast!=0.0))[0]
            contrast[i] = np.median(c_contrast[idx])

            contrast_err[i] = np.sqrt(np.var(c_contrast[idx])*len(idx)/np.double(len(idx)-1.))

        # Convert radii in pixels to arseconds:
        radii = radii*scale

        # Save results:
        fout = open(out_dir+'/contrast_curve_'+filename+'.dat','w')
        fout.write('# Radius ('') \t Magnitude Contrast \t Error\n')
        for i in range(len(radii)):
                    fout.write('{0: 3.3f} \t {1: 3.3f} \t {2: 3.3f} \n'.format(radii[i],\
                                                            contrast[i],contrast_err[i]))
        fout.close()

        # Plot final results to the user
        import matplotlib.pyplot as plt
        #plt.errorbar(radii,contrast,yerr=contrast_err)
        plt.plot(radii,contrast,color='black', label='Magnitude Contrast')
        plt.title(f'ShaneAO_{filename.split(".")[0]}')
        plt.xlabel('Distance from Centroid (arcsec)')
        plt.ylabel(r'$\Delta K_{s}$')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig(out_dir + filename.split('.')[0] + '.png', dpi=300)
        plt.show()

        fig, ax1 = plt.subplots(figsize=(5, 4))
        # print(np.where(radial_profile['col2'] > 0))
        ax1.plot(radii,contrast,color='black', label='Magnitude Contrast')
        ax1.set_ylim(6, 2)
        # ax1.set_xlim(-0.05, 1.5)
        ax2 = fig.add_axes([0.48, 0.45, 0.4, 0.4])
        # print(hdul[0].header)
        ax2.imshow(d, origin='lower')  # , cmap='RdGy_r', vmin=-1, vmax=1
        ax2.hlines(210, 255, 345, colors='w')
        ax2.text(333, 185, r"$3''$", ha='center', c='w')
        width = 96
        ax2.set_ylim(y0-width, y0+width)
        ax2.set_xlim(x0-width, x0+width)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.tick_params(axis='y', left=False)
        ax2.tick_params(axis='x', bottom=False)

        max_flux = np.max(d)
        print(np.where(d == 1.0))
        # dists_d_mag=np.zeros((2,256,256))
        # for i in range(256):
        #     for j in range(256):
        #         dist = np.sqrt((i-127) ** 2 + (j-127) ** 2) * 0.0183121
        #         d_mag = -2.5 * np.log10(hdul[0].data[i][j] / max_flux)
        #         dists_d_mag[0,i,j] = dist
        #         dists_d_mag[1,i,j] = d_mag
        #         ax1.plot(dist, d_mag, '.k', ms=2, alpha=0.2)
        # np.save(f'/home/tehan/Documents/GEMS/TOI-5344/figures/nessi.npy', dists_d_mag)

        ax1.set_ylabel(r'$\Delta$ Mag')
        ax1.set_xlabel(r'Angular Separation (")')
        plt.savefig(out_dir + filename.split('.')[0] + '_pub.png', dpi=300)
        plt.show()