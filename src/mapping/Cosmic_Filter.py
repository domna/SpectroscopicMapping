import pandas as pd
import numpy as np


def filter_where(arr, k):
    """
    Find values of an input (numpy.ndarray) that are greater than k.

    This function returns values of an input Numpy Array that are greater than
    a variable k.

    Parameters
    ----------
    arr : np.array
        Input data to search in
    k : float

    Returns
    -------
    np.array
        A 1D array consisting of values that fullfill the requirement "greater k".

    Examples
    --------
    test = np.array([[1, 3, 5, 7], [4, 5, 1, 6], [2, 3, 8, 1], [1, 3, 4, 1]])
    >>> filter_where(test, 6)
    array([7, 8])

    """
    return arr[np.where(arr > k)]


def filter_where_ind(arr, k):
    """
    Find indices of an input (numpy.ndarray) that are greater than k.

    This function returns indices of values of an input Numpy Array that are greater than
    a variable k.

    Parameters
    ----------
    arr : np.array
        Input data to search in
    k : float

    Returns
    -------
    np.array
        A 1D array consisting of 2 1D arrays with index pairs (x, y) of data points
        that fullfill the requirement "greater k".

    Examples
    --------
    test = np.array([[1, 3, 5, 7], [4, 5, 1, 6], [2, 3, 8, 1], [1, 3, 4, 1]])
    >>> filter_where_ind(test, 6)
    (array([0, 2], dtype=int64), array([3, 2], dtype=int64))

    """
    return np.where(arr > k)


def filt_surr(surr, perc, mean):
    """
    Simple filter to check whether the input surr is a signal.

    The function tests if the input value surr is above a modifiable mean level of the data.

    Parameters
    ----------
    surr : float
        Signal level of a data point.
    perc : float
        Modifier for mean level
    mean : float
        Mean value of several data points.

    Returns
    -------
    None or String
        Returns nothing if the point is on or beneath the mean level.
        Returns "signal" else.

    See Also
    --------
    The function is meant to filter for points that are possible cosmics. Those are
    above the noise of the data.

    Examples
    --------
    surr = 5
    mean = 4
    perc = 1.5
    >>> filt_surr(5, 1.5, 4)
    None
    surr = 5
    mean = 4
    perc = 1.0
    >>> filt_surr(5, 1.0, 4)
    "signal"
    """
    if surr > perc * mean:
        return "signal"


def it_list_var_2(place, wv, var):
    """
    Defines an iterator.

    The function tests where an index is and returns an iterator whith dimensions fitting
    it's whereabouts.

    Parameters
    ----------
    place : int
        Index number in informatic counting.
    wv : list
        len(List) should return the amount of indices.
    var : int
        Dividable by 2! Defines length of the iterator.

    Returns
    -------
    List of integers
        Returns a list of variable length.

    Examples
    --------
    wv = ["a", "b", "c", "d", "e", "f", "g", "h"]

    >>> it_list_var_2(3, wv, 4)
    [-2, -1, 0, 1, 2]

    >>> it_list_var_2(1, wv, 6)
    [-1, 0, 1, 2, 3]

    >>> it_list_var_2(8, wv, 2)
    [-1, 0]
    """
    iterator_i = list(range(-int(var / 2), int(var / 2) + 1))
    if (place - var / 2) < 0:
        return iterator_i[int(abs(place - var / 2)):]
    elif (place + var / 2) > (len(wv) - 1):
        return iterator_i[:int(len(iterator_i) - (place + var / 2 + 1 - len(wv)))]
    else:
        return iterator_i


def surr_var_loc(x_indic, y_indic, x_i, y_i, var=1):
    """
    Defines a list of index pairs (in numeric counting) around a point.

    The function checks where the point defined by x_indic and y_indic is and
    returns index pairs (in numeric counting) that describe the location of points located around it.
    The variable var defines how many points are included in the output.
    Points, that do not exist, are excluded from output.

    Parameters
    ----------
    x_indic, y_indic : int
        Index number in informatic counting of x, y.
    x_i, y_i : list
        len(List) should return the amount of indices in x or y direction.
    var : int
        Defines size of the output.

    Returns
    -------
    List
        Returns a list of variable length with pairs of indice numbers (numeric counting) for x and y.

    Examples
    --------
    test_x = ["ax", "bx", "cx", "dx", "ex"]
    test_y = ["ay", "by", "cy", "dy", "ey"]

    >>> surr_var_loc(4, 3, test_x, test_y, 1)
    [[4, 2], [3, 3], [4, 4]]

    >>> surr_var_loc(0, 0, test_x, test_y, 1)
    [[1, 0], [0, 1]]

    >>> surr_var_loc(2, 2, test_x, test_y, 2)
    [[2, 0], [1, 1], [2, 1], [3, 1], [0, 2], [1, 2], [3, 2], [4, 2], [1, 3], [2, 3], [3, 3], [2, 4]]
    """
    DOUT = []
    iterator_i_x = it_list_var_2(x_indic, x_i, 2 * var)
    iterator_i_y = it_list_var_2(y_indic, y_i, 2 * var)
    for i in iterator_i_y:
        for j in iterator_i_x:  # [abs(i): len(iterator_i_x) - abs(i)]:
            if abs(i) + abs(j) <= var:
                DOUT.append([int(x_indic + j), int(y_indic + i)])
    DOUT.remove([int(x_indic), int(y_indic)])
    return DOUT


def find_cosmics(dfin, wv_index, method_i="everything", mask="area", filter_length=8, filter_size=1, peak_to_noise=1.5,
                 signal_to_noise=1.3):
    """
    find_cosmics finds and smoothens Cosmics from data.

    The function takes a data frame (pandas package) with 4 dimensions as standard input.
    The amount of data checked and edited by the function can be adjusted with wv_index.
    method_i and mask allow to adjust the essential way this filter works. filter_length,
    filter size, signal_to_noise and peak_to_noise are parameters to optimize the filter
    for different kinds of data.

    Parameters
    ----------
    dfin : data frame
        Should contain the whole data measured. An edited copy of this
        is returned as an output.
    wv_index : int or list of int
        Index number (numeric counting) of the wavelength of our data_points.
        Can be given as a list, the program iterates over it.
    method_i : str
        Chooses filter method, e.g. "spectral" or "area".
    mask : str
        Method of flatten, spacial or spectral.
    filter_length : int
        Dividable by 2! Defines size of spectral width, which is checked for signals.
    filter_size : int
        Defines size of spacial area checked.
    peak_to_noise : float
        What do I define as a Cosmic? Cosmic > (mean) * signal_to_noise
    signal_to_noise: float
        What do I define as a signal? signal > (mean) * signal_to_noise

    Returns
    -------
    data frame
        Copy of input with filtered Cosmics

    See Also
    --------
    filter_where_ind : Find indices of an input (numpy.ndarray) that are greater than k.
    filt_area : Checks whether elements in points are signals or Cosmics.
    filt_spectral : Checks whether elements in points are signals or Cosmics.
    flatten : flatten consists of two different filters that can smooth Cosmics out.
    """

    dfout = dfin.copy()

    if method_i == "opt1" or method_i == "opt3":
        mean_datalist_i = list(dfin.mean(axis=0))

    if (filter_length % 2) != 0:
        raise ValueError("filter_length should be divisible by 2")

    if type(wv_index) == int:
        idx = pd.IndexSlice
        data_int = dfin.iloc[idx[:], wv_index]  # data at wavelength wv_index
        x = np.unique(data_int.index.get_level_values(0))  # indices of x-axis
        y = np.unique(data_int.index.get_level_values(1))  # indices of y-axis
        length_x_i = len(x)
        length_y_i = len(y)
        DIN_i = data_int.sort_index().values.reshape(length_x_i, length_y_i, order='C').astype(float)
        # reshape data into np array
        mean_DIN = np.mean(DIN_i)
        ind_max = filter_where_ind(DIN_i,
                                   peak_to_noise * mean_DIN)  # indices of data points greater than peak_to_noise*mean

        if method_i == "area":
            DOUT = filt_area(ind_max, DIN_i, x, y, signal_to_noise, filter_size)
            dfout_i = flatten(DOUT, dfout, wv_index, "area", DIN_i, x, y)

        elif method_i == "spectral":
            DOUT = filt_spectral(ind_max, dfin, wv_index, x, y, filter_length)
            dfout_i = flatten(DOUT, dfout, wv_index, "spectral", DIN_i, x, y, filter_length)

        elif method_i == "everything":
            DOUT = filt_area(ind_max, DIN_i, x, y, signal_to_noise, filter_size)
            DOUT = filt_spectral(DOUT, dfin, wv_index, x, y, filter_length)
            dfout_i = flatten(DOUT, dfout, wv_index, mask, DIN_i, x, y, filter_length, filter_size)

        elif method_i == "opt1":
            DOUT = filt_area(ind_max, DIN_i, x, y, signal_to_noise, filter_size)
            DOUT = filt_spectral_opt1(DOUT, dfin, wv_index, x, y, mean_datalist_i, filter_length)
            dfout_i = flatten(DOUT, dfout, wv_index, mask, DIN_i, x, y, filter_length, filter_size)

        elif method_i == "opt3":
            DOUT = filt_area(ind_max, DIN_i, x, y, signal_to_noise, filter_size)
            DOUT = filt_spectral_opt3(DOUT, dfin, wv_index, x, y, mean_datalist_i, filter_length)
            dfout_i = flatten(DOUT, dfout, wv_index, mask, DIN_i, x, y, filter_length, filter_size)

        else:
            raise ValueError("Method not found")

    elif type(wv_index) == list:
        for index in wv_index:
            idx = pd.IndexSlice
            data_int = dfin.iloc[idx[:], index]  # data at wavelength wv_index
            x = np.unique(data_int.index.get_level_values(0))  # indices of x-axis
            y = np.unique(data_int.index.get_level_values(1))  # indices of y-axis
            length_x_i = len(x)
            length_y_i = len(y)
            DIN_i = data_int.sort_index().values.reshape(length_x_i, length_y_i, order='C').astype(float)
            # reshape data into np array
            mean_DIN = np.mean(DIN_i)
            ind_max = filter_where_ind(DIN_i,
                                       peak_to_noise * mean_DIN)
            # indices of data points greater than peak_to_noise*mean

            if method_i == "area":
                DOUT = filt_area(ind_max, DIN_i, x, y, signal_to_noise, filter_size)
                dfout_i = flatten(DOUT, dfout, index, "area", DIN_i, x, y)

            elif method_i == "spectral":
                DOUT = filt_spectral(ind_max, dfin, index, x, y, filter_length)
                dfout_i = flatten(DOUT, dfout, index, "spectral", DIN_i, x, y, filter_length)

            elif method_i == "everything":
                DOUT = filt_area(ind_max, DIN_i, x, y, signal_to_noise, filter_size)
                DOUT = filt_spectral(DOUT, dfin, index, x, y, filter_length)
                dfout_i = flatten(DOUT, dfout, index, mask, DIN_i, x, y, filter_length, filter_size)

            elif method_i == "opt1":
                DOUT = filt_area(ind_max, DIN_i, x, y, signal_to_noise, filter_size)
                DOUT = filt_spectral_opt1(DOUT, dfin, index, x, y, mean_datalist_i, filter_length)
                dfout_i = flatten(DOUT, dfout, index, mask, DIN_i, x, y, filter_length, filter_size)

            elif method_i == "opt3":
                DOUT = filt_area(ind_max, DIN_i, x, y, signal_to_noise, filter_size)
                DOUT = filt_spectral_opt3(DOUT, dfin, index, x, y, mean_datalist_i, filter_length)
                dfout_i = flatten(DOUT, dfout, index, mask, DIN_i, x, y, filter_length, filter_size)

            else:
                raise ValueError("method not found")

    else:
        raise ValueError("Error: incorrect \"wv_index\" given")

    return dfout_i


def filt_area(points, DIN, x_i, y_i, signal_to_noise=1.3, var=1):
    """
    Checks whether elements in points are signals or Cosmics.

    The function creates a list of surrounding points (in space regime) of the elements
    in the list "points" and searches for signals in that list. Data points with signal
    in the vicinity are less likely to be Cosmics. Those with enough signal in
    the vicinity are taken out of the list.
    Output and input should have the exact same format otherwise filters cannot be
    switched anymore.

    Parameters
    ----------
    points : list
        List of possible cosmics in the data.
        Points are supposed to be index numbers in numeric counting
        rather than values of x and y.
    x_i, y_i : list
        len(List) should return the amount of indices in x or y direction.
    signal_to_noise : float
        What do I define as signal? signal > (mean) * signal_to_noise
    var : int
        Defines size of the area checked.

    Returns
    -------
    List = [[x indices][y indices]]
        Returns a list of 2 lists. Those two lists contain the x and y
        index numbers in numeric counting.

    See Also
    --------
    surr_var_loc : Defines a list of index pairs (in numeric counting) around a point.
    filt_surr : Simple filter to check whether the input surr is a signal.
    """
    DxOUT = []
    DyOUT = []
    mean_DIN = np.mean(DIN)

    for i in range(0, len(points[0])):
        x_ind = points[0][i]
        y_ind = points[1][i]
        surr = surr_var_loc(x_ind, y_ind, x_i, y_i, var)
        reff_check = []

        for j in range(0, len(surr)):
            reff_check.append(filt_surr(DIN[surr[j][0]][surr[j][1]], signal_to_noise, mean_DIN))

        if "signal" not in reff_check:
            DxOUT.append(x_ind)
            DyOUT.append(y_ind)

    return [DxOUT, DyOUT]


def filt_spectral(points, DIN, wv_index, x_i, y_i, var_i=8,
                  signal_to_noise=1.3):  # points should be the values of x and y rather than indices
    """
    Checks whether elements in points are signals or Cosmics.

    The function creates a list of surrounding points (spectral) of the elements
    in the list "points" and searches for signals in that list. Data points with signal
    in the vicinity are less likely to be Cosmics. Those with enough signal in
    the vicinity are taken out of the list.
    Output and input should have the exact same format otherwise filt_area
    and filt_spectral cannot be switched anymore.

    Parameters
    ----------
    points : list
        List of possible cosmics in the data.
        Points are supposed to be index numbers in numeric counting
        rather than values of x and y.
    DIN : data frame
        The spectra of the corresponding data points are extracted from DIN.
    wv_index : int
        Index number (numeric counting) of the wavelength of our data_points.
    x_i, y_i : list
        List of all indexes of x and y direction.
    var_i : int
        Dividable by 2! Defines size of spectral width, which is checked for signals.
    signal_to_noise : float
        What do I define as signal? signal > (mean) * signal_to_noise

    Returns
    -------
    List = [[x indices][y indices]]
        Returns a list of 2 lists. Those two lists contain the x and y
        index numbers in numeric counting.

    See Also
    --------
    it_list_var_2 : The function tests where an index is and returns an iterator with
                    dimensions fitting it's whereabouts.
    filt_surr : Simple filter to check whether the input surr is a signal.

    Would be nice to just use iloc right? Nope iloc with MultiIndices works miserable...
    x_ind = x_i[points[0][i]]
    y_ind = y_i[points[1][i]]
    wv_data = DIN.loc[(x_ind, y_ind)]
    """

    DxOUT = []
    DyOUT = []
    for i in range(0, len(points[0])):
        x_ind = x_i[points[0][i]]
        y_ind = y_i[points[1][i]]
        wv_data = DIN.loc[(x_ind, y_ind)]
        wv_data_mean = wv_data.mean()
        reff_check = []
        iterator = it_list_var_2(wv_index, wv_data, var_i)
        iterator.remove(0)
        for j in iterator:
            reff_check.append(filt_surr(wv_data.iloc[wv_index + j], signal_to_noise, wv_data_mean))
        if reff_check.count("signal") < len(iterator) / 2:
            DxOUT.append(points[0][i])
            DyOUT.append(points[1][i])

    return [DxOUT, DyOUT]


def filt_spectral_opt1(points, DIN, wv_index, x_i, y_i, mean_datalist, var_i=8,
                       signal_to_noise=1.3):  # points should contain the values of x and y rather than indices
    """
    Checks whether elements in points are signals or Cosmics.
    Works equivalent to filt_spectral with other mean values.

    The function creates a list of surrounding points (spectral) of the elements
    in the list points and searches for signals in that list. Points with signal
    in the vicinity are less likely to be Cosmics. Those with enough signal in
    the vicinity are taken out of the list.
    Output and input should have the exact same format otherwise filters cannot
    be switched anymore.

    Parameters
    ----------
    points : list
        List of possible cosmics in the data.
        Points are supposed to be index numbers in numeric counting
        rather than values of x and y.
    DIN : data frame
        The spectrum of the corresponding data points is extracted from DIN.
    wv_index : int
        Index number (numeric counting) of the wavelength of our data_points.
    x_i, y_i : list
        List of all indexes of x and y direction.
    mean_datalist : list
        Contains wavelength specific mean values.
    signal_to_noise : float
        What do I define as signal? signal > (mean) * signal_to_noise
    var_i : int
        Dividable by 2! Defines size of spectral width, which is checked for signals.

    Returns
    -------
    List = [[x indices][y indices]]
        Returns a list of 2 lists. Those two lists contain the x and y
        index numbers in numeric counting.

    See Also
    --------
    it_list_var_2 : The function tests where an index is and returns an iterator whith
                    dimensions fitting it's whereabouts.
    filt_surr : Simple filter to check whether the input surr is a signal.
    filt_spectral: Checks whether elements in points are signals or Cosmics.

    Examples
    --------
    """
    DxOUT = []
    DyOUT = []
    for i in range(0, len(points[0])):
        x_ind = x_i[points[0][i]]
        y_ind = y_i[points[1][i]]
        wv_data = DIN.loc[(x_ind, y_ind)]
        reff_check = []
        iterator = it_list_var_2(wv_index, wv_data, var_i)
        iterator.remove(0)
        for j in iterator:
            reff_check.append(filt_surr(wv_data.iloc[wv_index + j], signal_to_noise, mean_datalist[wv_index + j]))
        if reff_check.count("signal") < len(iterator) / 2:
            DxOUT.append(points[0][i])
            DyOUT.append(points[1][i])
    return [DxOUT, DyOUT]


def filt_spectral_opt3(points, DIN, wv_index, x_i, y_i, mean_datalist, var_i=8,
                       signal_to_noise=1.3):  # points should be the values of x and y rather than indices
    """
    Checks whether elements in points are signals or Cosmics.

    A fraction of all cosmics is broad in the spectral regime and seems
    indistinguishable from other signals (that are not supposed to be filtered).
    This function offers a strong and fast solution to filter even those cosmics.

    The function creates a list of points surrounding the elements (in the spectral regime)
    in the list "points" and searches for signals in that list. If signal is found, the
    spacial area around it is checked. The condition is that every signal is supposed to
    have signal in more than one location in at least one wavelength around it.

    Parameters
    ----------
    points : list
        List of possible cosmics in the data.
        Points are supposed to be index numbers in numeric counting
        rather than values of x and y.
    DIN : data frame
        The spectrum of the corresponding data points is extracted from DIN.
    wv_index : int
        Index number (numeric counting) of the wavelength of our data_points.
    x_i, y_i : list
        List of all indexes of x and y direction.
    mean_datalist : list
        Contains wavelength specific mean values.
    var_i : int
        Dividable by 2! Defines size of spectral width, which is checked for signals.
    signal_to_noise : float
        What do I define as signal? signal > (mean) * signal_to_noise

    Returns
    -------
    List = [[x indices][y indices]]
        Returns a list of 2 lists. Those two lists contain the x and y
        index numbers in numeric counting.

    See Also
    --------
    it_list_var_2 : The function tests where an index is and returns an iterator with
                    dimensions fitting it's whereabouts.
    filt_surr : Simple filter to check whether the input surr is a signal.
    filt_spectral : Checks whether elements in points are signals or Cosmics.
    filt_area : Checks whether elements in points are signals or Cosmics.
    """

    DxOUT = []
    DyOUT = []

    for i in range(0, len(points[0])):
        x_ind = x_i[points[0][i]]
        y_ind = y_i[points[1][i]]
        surr = surr_var_loc(points[0][i], points[1][i], x_i, y_i)
        wv_data = DIN.loc[(x_ind, y_ind)]
        reff_check = []
        reff_check_i = []
        iterator = it_list_var_2(wv_index, wv_data, var_i)
        iterator.remove(0)
        for j in iterator:
            if filt_surr(wv_data.iloc[wv_index + j], signal_to_noise, mean_datalist[wv_index + j]) == "signal":
                d_array = DIN.iloc[pd.IndexSlice[:], wv_index + j].sort_index().values.reshape(len(x_i), len(y_i),
                                                                                               order='C').astype(float)
                for k in range(0, len(surr)):
                    reff_check_i.append(
                        filt_surr(d_array[surr[k][0]][surr[k][1]], signal_to_noise, mean_datalist[wv_index + j]))
                if "signal" in reff_check_i:
                    reff_check.append("signal")
        if reff_check.count("signal") < 1:
            DxOUT.append(points[0][i])
            DyOUT.append(points[1][i])

    return [DxOUT, DyOUT]


def flatten(points, dfout, wv_index, method, DIN, x_i, y_i, var_i=8, var=1):
    """
    Flatten offers two different methods to eradicate those pesky little Cosmics.

    The function consists of two different filters that smooth the Cosmics out.
    The methods spectral and area create a list of points in the vicinity of the Cosmic and
    average over their values. The result is given back as the new value of the Cosmic.

    The area method usually smooths a lot rougher than spectral regarding actual Cosmics.

    Parameters
    ----------
    points : list
        List of possible cosmics in the data.
        Points are supposed to be index numbers in numeric counting
        rather than values of x and y.
    dfin : data frame
        Should contain the whole data measured. An edited copy of this
        is returned as an output.
    dfout : data frame
        Copy of dfin in.
    wv_index : int
        Index number (numeric counting) of the wavelength of our data_points.
    method : str
        Chooses filter type, e.g. "spectral" or "area".
    x_i, y_i : list
        List of all indexes of x and y direction.
    var_i : int
        Dividable by 2! Defines size of spectral width, which is checked for signals.

    Returns
    -------
    data frame
        An edited copy of dfin is returned as an output. Cosmics are smoothed.

    See Also
    --------
    it_list_var_2 : The function tests where an index is and returns an iterator with
                    dimensions fitting it's whereabouts.
    surr_var_loc : Defines a list of index pairs (in numeric counting) around a point.
    filt_surr : Simple filter to check whether the input surr is a signal.

    Examples
    --------
    """
    if method == "spectral":
        for i in range(0, len(points[0])):
            x_ind = x_i[points[0][i]]
            y_ind = y_i[points[1][i]]
            wv_data = dfout.loc[(x_ind, y_ind)]
            iterator = it_list_var_2(wv_index, wv_data, var_i)
            iterator.remove(0)
            avg_surr4 = 0
            for k in iterator:
                avg_surr4 += wv_data.iloc[wv_index + k]
            dfout.loc[(x_ind, y_ind), list(dfout.columns.values)[wv_index]] = avg_surr4 / len(iterator)

    elif method == "area":
        for i in range(0, len(points[0])):
            x_ind = x_i[points[0][i]]
            y_ind = y_i[points[1][i]]
            surr = surr_var_loc(points[0][i], points[1][i], x_i, y_i, var)
            avg_surr4 = 0
            for j in range(0, len(surr)):
                avg_surr4 += DIN[surr[j][0]][surr[j][1]]
            dfout.loc[(x_ind, y_ind), list(dfout.columns.values)[wv_index]] = avg_surr4 / (len(surr))

    return dfout
