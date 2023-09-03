# -*- coding: utf-8 -*-
# +
import os
import glob
import numpy as np
import rasterio

class atmosCorr(object):
    
    '''Atmospheric Correction in Optical domain'''
    
    def __init__(self, path, nodata = -99999):
        
        '''
        
        Parameter:
        
            path: String. The folder in which the satellite bands are located. This images could be Landsat
                  Collection 2 Level-1. For example: path = r'/folder/image/raster'.
            
            nodata: The NoData value to replace with -99999.
        '''
        
        self.path = path
        self.nodata = nodata
        
        path = path.replace(os.sep, '/')
        
        path = os.chdir(path)
        
        file_mtl = glob.glob("*_MTL.txt")

        dict_mtl = {}

        with open(file_mtl[0], 'r') as mtl:
            for lines in mtl:
                lines = lines.strip()
                if lines != 'END':
                    key, value = lines.split('=')
                    dict_mtl[key] = value

        names_bands = glob.glob("*B?*.TIF")
        names_bands.sort()
        
        self.dict_mtl = dict_mtl
        self.names_bands = names_bands

    def RAD(self, sat = 'LC08'):
        
        '''
        Conversion to TOA Radiance. Landsat Level-1 data can be converted to TOA spectral radiance 
        using the radiance rescaling factors in the MTL file:
        
        Lλ = MLQcal + AL 
        
        where:

        Lλ = TOA spectral radiance (Watts/(m2*srad*μm))
        ML = Band-specific multiplicative rescaling factor from the metadata (RADIANCE_MULT_BAND_x, where x is the band number)
        AL = Band-specific additive rescaling factor from the metadata (RADIANCE_ADD_BAND_x, where x is the band number)
        Qcal =  Quantized and calibrated standard product pixel values (DN) 

        Parameters:
            sat: Type of Satellite. It could be Landsat-5 TM, Landsat-8 OLI or Landsat-9 OLI-2.
        
        Return: 
            An array with radiance values with 3d, i.e. (rows, cols, bands).
        '''
        
        if sat == 'LC09':
            # Landsat-9 bands -> blue, green, red, nir, swir1 and swir2
            bands_lc09 = ['B2.','B3.','B4.','B5.','B6.','B7.']
            output_names = [name for name in self.names_bands if (name[41:44] in bands_lc09)]
        
        if sat == 'LC08':
            # Landsat-8 bands -> blue, green, red, nir, swir1 and swir2
            bands_lc08 = ['B2.','B3.','B4.','B5.','B6.','B7.']
            output_names = [name for name in self.names_bands if (name[41:44] in bands_lc08)]
        
        if sat == 'LT05':
            # Landsat-5 bands -> blue, green, red, nir, swir1 and swir2
            bands_lt05 = ['B1.','B2.','B3.','B4.','B5.','B7.']
            output_names = [name for name in self.names_bands if (name[41:44] in bands_lt05)]

        list_bands = []
        for i in output_names:
            with rasterio.open(i, 'r') as b:
                list_bands.append(b.read(1))

        dn = np.stack(list_bands)

        # data in [rows, cols, bands]
        #dn = np.moveaxis(st, 0, -1) 
        
        rows = dn.shape[1]
        
        cols = dn.shape[2]

        bands = dn.shape[0]

        # nodata
        if np.isnan(np.sum(dn)):
            dn[np.isnan(dn)] = self.nodata
        
        tetha = float(self.dict_mtl['SUN_ELEVATION '])
        
        rad_bands = []
        
        for i in range(bands):
            
            ML = float(self.dict_mtl['RADIANCE_MULT_BAND_' + str(i+1) + ' '])
            
            AL = float(self.dict_mtl['RADIANCE_ADD_BAND_' + str(i+1) + ' '])
            
            radiance = np.add(np.multiply(dn[i,:,:], ML), AL) # Lλ = ML*DN + AL 
            
            rad_bands.append(radiance)
        
        arr_rad = np.moveaxis(np.stack(rad_bands), 0, -1) 
        
        # negative values -> nodata
        arr_rad[arr_rad < 0] = np.nan
        
        return arr_rad
    
    def TOA(self, sat = 'LC08'):
        
        '''
        A reduction in scene-to-scene variability can be achieved by converting the at-sensor 
        spectral radiance to exoatmospheric TOA reflectance, also known as in-band planetary albedo.
        
        Equation to obtain TOA reflectance:
        
        ρλ′ = Mρ*DN + Aρ
        
        ρλ = ρλ′/sin(theta)
        
        Parameters:
            sat: Type of Satellite. It could be Landsat-5 TM, Landsat-8 OLI or Landsat-9 OLI-2.
        
        Return: 
            An array with TOA values with 3d, i.e. (rows, cols, bands).
        '''
        
        if sat == 'LC09':
            # Landsat-9 bands -> blue, green, red, nir, swir1 and swir2
            bands_lc09 = ['B2.','B3.','B4.','B5.','B6.','B7.']
            output_names = [name for name in self.names_bands if (name[41:44] in bands_lc09)]
        
        if sat == 'LC08':
            # Landsat-8 bands -> blue, green, red, nir, swir1 and swir2
            bands_lc08 = ['B2.','B3.','B4.','B5.','B6.','B7.']
            output_names = [name for name in self.names_bands if (name[41:44] in bands_lc08)]
        
        if sat == 'LT05':
            # Landsat-5 bands -> blue, green, red, nir, swir1 and swir2
            bands_lt05 = ['B1.','B2.','B3.','B4.','B5.','B7.']
            output_names = [name for name in self.names_bands if (name[41:44] in bands_lt05)]

        list_bands = []
        for i in output_names:
            with rasterio.open(i, 'r') as b:
                list_bands.append(b.read(1))

        dn = np.stack(list_bands)

        # data in [rows, cols, bands]
        #dn = np.moveaxis(st, 0, -1) 
        
        rows = dn.shape[1]
        
        cols = dn.shape[2]

        bands = dn.shape[0]

        # nodata
        if np.isnan(np.sum(dn)):
            dn[np.isnan(dn)] = self.nodata
        
        tetha = float(self.dict_mtl['SUN_ELEVATION '])
        
        toa_bands = []
        
        for i in range(bands):
            
            Mp = float(self.dict_mtl['REFLECTANCE_MULT_BAND_' + str(i+1) + ' '])
            
            Ap = float(self.dict_mtl['REFLECTANCE_ADD_BAND_' + str(i+1) + ' '])
            
            plambda = np.add(np.multiply(dn[i,:,:], Mp), Ap) # ρλ′ = Mρ*DN + Aρ
            
            TOA = plambda/np.sin((tetha*np.pi/180)) # ρλ = ρλ′/sin(theta)
            
            toa_bands.append(TOA)
        
        arr_toa = np.moveaxis(np.stack(toa_bands), 0, -1) 
        
        # negative values -> nodata
        arr_toa[arr_toa < 0] = np.nan
        
        return arr_toa
    
    def DOS(self, sat = 'LC08', mindn = None):
        
        '''
        The Dark Object Subtraction Method was proposed by Chavez (1988). This image-based 
        atmospheric correction method considers absolutely critical and valid the existence 
        of a dark object in the scene, which is used in the selection of a minimum value in 
        the haze correction. The most valid dark objects in this kind of correction are areas 
        totally shaded or otherwise areas representing dark water bodies.
        
        Parameters:
        
            sat: Type of Satellite. It could be Landsat-5 TM, Landsat-8 OLI or Landsat-9 OLI-2.
            
            mindn: Min of digital number for each band in a list.
            
        Return: 
            An array with Surface Reflectance values with 3d, i.e. (rows, cols, bands).
        
        References:
        
            Chavez, P.S. (1988). An Improved Dark-Object Subtraction Technique for Atmospheric 
            Scattering Correction of Multispectral Data. Remote Sensing of Envrironment, 24(3), 459-479.
        '''
        
        if sat == 'LC09':
            bands_lc09 = ['B2.','B3.','B4.','B5.','B6.','B7.']
            output_names = [name for name in self.names_bands if (name[41:44] in bands_lc09)]
            # ESUN for Landsat-9 bands -> blue, green, red, nir, swir1 and swir2
            # https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
            ESUN = [2067, 1893, 1603, 972.6, 245.0, 79.72]
            
        if sat == 'LC08':
            bands_lc08 = ['B2.','B3.','B4.','B5.','B6.','B7.']
            output_names = [name for name in self.names_bands if (name[41:44] in bands_lc08)]
            # ESUN for Landsat-8 bands -> blue, green, red, nir, swir1 and swir2
            # https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
            ESUN = [2067, 1893, 1603, 972.6, 245.0, 79.72]

        if sat == 'LT05':
            # Landsat-5 bands -> blue, green, red, nir, swir1 and swir2
            bands_lt05 = ['B1.','B2.','B3.','B4.','B5.','B7.']
            output_names = [name for name in self.names_bands if (name[41:44] in bands_lt05)]
            
        list_bands = []
        for i in output_names:
            with rasterio.open(i, 'r') as b:
                list_bands.append(b.read(1))

        dn = np.stack(list_bands)

        # data in [rows, cols, bands]
        #dn = np.moveaxis(st, 0, -1) 
        
        rows = dn.shape[1]
        
        cols = dn.shape[2]

        bands = dn.shape[0]

        # Min of digital number for each band
        if mindn is None:
            dn_mins = [np.nanmin(dn[i,:,:][np.nonzero(dn[i,:,:])]) for i in range(bands)]
        else:
            dn_mins = mindn
            
            if not len(mindn) == bands:
                raise ValueError(f'The length of the "mindn" argument must be equal to the number'
                                f'of bands. Length of "mindn" is {len(mindn)}, and length of bands'
                                f'is {bands}')
                
        # nodata
        if np.isnan(np.sum(dn)):
            dn[np.isnan(dn)] = self.nodata
        
        tetha = float(self.dict_mtl['SUN_ELEVATION '])
        u_a = float(self.dict_mtl['EARTH_SUN_DISTANCE '])
        
        dos_bands = []
        
        for i in range(bands):
            
            ML = float(self.dict_mtl['RADIANCE_MULT_BAND_' + str(i+1) + ' '])
            
            AL = float(self.dict_mtl['RADIANCE_ADD_BAND_' + str(i+1) + ' '])
            
            L = np.add(np.multiply(dn[i,:,:], ML), AL) # L = ML*DN+AL
            
            L_min = np.add(np.multiply(dn_mins[i], ML), AL) # for each band
            
            DOS = np.pi*(np.multiply((L - L_min), (u_a)**2))/(ESUN[i]*np.sin((tetha*np.pi/180)))
            
            dos_bands.append(DOS)
        
        arr_dos = np.moveaxis(np.stack(dos_bands), 0, -1) 
        
        # negative values -> nodata
        arr_dos[arr_dos < 0] = np.nan
        
        return arr_dos
