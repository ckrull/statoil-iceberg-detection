import itertools
from math import factorial

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from scipy.stats import kurtosis, skew
from sklearn.metrics import (accuracy_score, precision_score, r2_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import guidedfilter as gf


# from SciPy cookbook http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """


    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def is_sequence(arg):
    """ check if a given object is a sequence e.g. list """
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))
    
# Data preprocessing funtions for Statoil Kaggle competition

def filter_guided(img, r = 2, eps = 10):
    """Apply a guided filter to img data
    
    Parameters
    ----------
    img : array_like
        The img data to be filtered. Can be a square image or flattened then a square size is assumed.  
    r : int, optional
        Guided filter window radius (the default is 2)
    eps : int, optional
        guided filter regularization (roughly, allowable variance of non-edge noise) (the default is 10)
    
    Returns
    -------
    array_like
        the filtered image
    """

    size=int(np.sqrt(len(img)))
    img=img.reshape(size,size)
    denoise = gf.guided_filter(img, img, r, eps)
    return denoise
 
def filter_lee(img, size=7): 
    """Apply a lee speckle removal filter (radar processing) to img data.
    
    Parameters
    ----------
    img : array_like
        the input image
    size : int, optional
        Lee window size (the default is 7)
    
    Returns
    -------
    array_like
        the filtered image
    References
    ----------
    .. [1] Lee, J.-S. Speckle analysis and smoothing of synthetic aperture radar images. Computer Graphics and Image Processing 17, 24–32 (1981).

    """

    size=int(np.sqrt(len(img)))
    img = img.reshape(size,size)
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = variance(img)
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def scaling_function(img):
    """Apply a min max scaling to the input data
    
    Parameters
    ----------
    img : array_like
        the input image can be any shape
    Returns
    -------
    array_like
        the scaled image, has the same shape as the input
    """

    data= img.flatten()
    data -=np.min(data)
    data /=np.max(data)
    return data.reshape(*img.shape)
    

def prepare_data(df,filter_function=filter_guided,img_size=75,dim=[0,1,2],scale= 'img',rnd=False):
    """ Data preparation pipeline for the Statoil competition.

    
    Parameters
    ----------
    df : pandas.DataFrame
        the colums  band_1, and band_2 will be used
    filter_function : function, optional        
        any function to filter each image with need to take an array_like image as input (the default is filter_guided, which is a guided filter)
    img_size : int, optional
        width and height of the square image data (the default is 75)
    dim : list, optional
        lsitto select the desired channels, possible values are 0,1,2 (the default is [0,1,2], which are all three channels HH, HV, HV-HH)
    scale : str, optional
        determines how the data is scaled:
        'img' : each image is min/max scaled individually
        'batch' : the complete batch is min/max scaled per channel
        None: no scaling  (the default is 'img')
    rnd : bool, optional
        return random data in the same shape as the input (the default is False)
    
    Returns
    -------
    array_like  
        the image data in array form. the shape is(size,size,channels)
    """

    
    length=len(df)
    dim_len=len(dim)
    data=[]
    b1 = np.vstack(df['band_1'])
    b2 = np.vstack(df['band_2'])
    b3 = b2-b1
    if rnd:
        b1 = np.random.rand(*b1.shape)
        b2 = np.random.rand(*b1.shape)
        b3 = np.random.rand(*b1.shape)     
    for d in dim:
        a=[b1,b2,b3][d]
        data.append(a)
    if scale == 'batch': 
        scalers = [MinMaxScaler(copy=False) for x in range(dim_len)]
        for idx,scaler in enumerate(scalers):
            scaler.fit_transform(data[idx])
    if scale == 'img':
        data = [np.apply_along_axis(scaling_function,1,b) for b in data ]         
    if filter_function:
        data = [np.apply_along_axis(filter_function,1,b) for b in data ]    
    X = np.hstack(data).reshape(length,dim_len,img_size,img_size)
    return np.moveaxis(X,1,-1)


def extract_moments(df):
    """helper function add the statical moments to a statoil dataframe
    
    Parameters
    ----------
    df : pandas.DataFrame
        the input DataFrame
    Returns
    -------
    pandas.DataFrame    
        the output DataFrame with the added moments
    """

    df['band_1_mean'] = np.array([np.mean(x) for x in df['band_1']])
    df['band_1_var'] = np.array([np.var(x) for x in df['band_1']])
    df['band_1_skew'] = np.array([skew(x) for x in df['band_1']])
    df['band_1_kurt'] = np.array([kurtosis(x) for x in df['band_1']])
    df['band_2_skew'] = np.array([skew(x) for x in df['band_2']])
    df['band_2_kurt'] = np.array([kurtosis(x) for x in df['band_2']])
    df['band_2_mean'] = np.array([np.mean(x) for x in df['band_2']])
    df['band_2_var'] = np.array([np.var(x) for x in df['band_2']])
    return df

def plot_training_history(data,model_name=None,smooth_window=None,smooth_order=1):
    if not smooth_window:
        smooth_window= len(data['val_loss'])/100
        if smooth_window % 2 == 0:
            smooth_window +=1
        if smooth_window < 3:
            smooth_window = 3
        print(f'Using a smooth window of {smooth_window}')
    best_val=np.min(data['val_loss'])
    best_epoch =np.argmin(data['val_loss'])
    _, ax = plt.subplots(figsize=(7,5))
    ax.plot(data['loss'],'.',color='deepskyblue' )
    ax.plot(data['val_loss'],'.' ,color='orange')
    ax.legend(['Training Loss','Validation Loss'])
    ax.plot(savitzky_golay(data['loss'],smooth_window,smooth_order), color='darkslategray')
    ax.plot(savitzky_golay(data['val_loss'],smooth_window,smooth_order), color='sienna') 
    ax.annotate('best' , xy=(best_epoch, best_val), xytext=(best_epoch, best_val-0.05),
            arrowprops=dict(facecolor='black', shrink=0.01,width=1),)
    ax.annotate('best loss:'+ str(round(best_val,4))+' epoch:'+ str(round(best_epoch,4)),
                xy=(0.05, 0.05), xycoords='axes fraction')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    titlestr= 'Training: Loss '
    if model_name:
        titlestr += model_name
    plt.title(titlestr)
    plt.show()


def dict_product(dicts):
    """return a product of dictionaries
    
    Parameters
    ----------
    dicts : list
        A list of dictionaries
    Returns
    -------
    generator
        creates dictionaries that are a product of the input dict

    Example
    -------
    list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """

    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
