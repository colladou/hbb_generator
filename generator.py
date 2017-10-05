from __future__ import print_function

import numpy as np
import h5py 

# hard code get_variable_names
# check that in functions I am returning the transformed data and not transforming the original data
# implement management for trailing samples in batch size
# extract weight and label

def get_variable_names(set_name):
    """
    Returns a dictionary dataset_name: variable_names with the variables to extract from each dataset of the hdf5 file.
    Here is where variable modification must be done and the list is hard coded.
    Variable and dataset names must match the ones in the named hdf5 file.
    """
    # here we hard code the names of the variables we want to use and from which set they come
    var_names = {}
    if set_name == 'hl_tracks':
        var_names['jets'] = ('pt', 'eta')
        var_names['subjet1'] = ('pt', 'eta', 
                              'ip3d_ntrk', 'ip2d_pu', 'ip2d_pc', 'ip2d_pb', 'ip3d_pu', 'ip3d_pc', 'ip3d_pb',
                              'mu_dR', 'mu_mombalsignif', 'mu_d0', 'mu_pTrel', 'mu_qOverPratio', 'mu_scatneighsignif',
                              'jf_dr', 'jf_efc', 'jf_m', 'jf_n2t', 'jf_ntrkAtVx', 'jf_nvtx', 'jf_nvtx1t', 'jf_sig3d', 'jf_deta', 'jf_dphi',
                              'sv1_dR', 'sv1_efc', 'sv1_Lxyz', 'sv1_Lxy', 'sv1_m', 'sv1_n2t', 'sv1_ntrkv', 'sv1_normdist',
                              'dphi_fatjet', 'deta_fatjet', 'dr_fatjet',
                              'mask')
        var_names['subjet2'] = var_names['subjet1']
        merge_order = ('jets', 'subjet1', 'subjet2')
    elif set_name == 'tracks':
        pass
    else:
        print(set_name)        
        raise NotImplementedError
    return [var_names, merge_order]

def load_mean_and_std(set_name, load_path="./"):
    """
    Returns mean and std vectors for scaling and centering purposes
    set_name: Name of the set to use, for example 'hl_tracks'
    load_path: Path where the vectors are located.
    """
    #mean_vector = np.load(load_path + "%s_mean_vector.npy" % set_name)
    #std_vector = np.load(load_path + "%s_std_vector.npy" % set_name)
    mean_vector, std_vector = 1, 1
    std_vector[std_vector == 0] = 1  # prevent x/0 division
    assert np.sum(np.is_nan(mean_vector)) == 0, "Nan value found in mean vector"
    assert np.sum(np.is_nan(std_vector)) == 0, "Nan value found in std vector"
    return [mean_vector, std_vector]

def flatten(data, name_list):
    """ 
    Flattens a named numpy array so it can be used with pure numpy.
    Input: named array and list of names of variables in the named array
    Output: numpy array

    Example;
    print(flatten(sample_jets[['pt', 'eta']]))
    """
    ftype = [(name, float) for name in name_list]
    flat = data.astype(ftype).view(float).reshape(data.shape + (-1,))
    return flat.swapaxes(1, len(data.shape))

def extract_variables(data, variable_names):    
    return data[variable_names][:]

def get_num_samples(data_file, axis=0, dataset_name=""):
    """
    Returns the length of an axis of a dataset of a hdf5 file.
    By default it uses the first axis and first dataset. 
    data_file: h5py open file
    """
    if dataset_name == "":
        data_sets = list(data_file.keys())
        assert len(data_sets) > 0, "There must be at least one dataset on the hdf5 file"
        data = data_file.get(data_sets[0])
    else:
        data = data_file.get(dataset_name)
    return data.shape[axis]

def merge_batches_from_categories(merge_list):
    """
    Returns a numpy array with the horizontal stacking of the arrays in merge_list
    merge_list: List of numpy arrays with the same size on the first axis
    """
    data_batch = None
    for category_batch in merge_list:
        if data_batch is None:
            data_batch = category_batch
        else:
            assert data_batch.shape[0] == category_batch.shape[0], "Numpy arrays in list must have the same number of samples"
            data_batch = hstack((data_batch, category_batch))
    return data_batch

def scale_and_center(data, mean_vector, std_vector):
    """
    Returns scaled and centered dataset
    data: Numpy array with the data.
    mean_vector: Numpy array with the mean for each feature
    std_vector: Numpy array with the standard deviation for each feature
    """
    return (data - mean_vector)/std_vector

def concatenate_names_from_categories(var_names, merge_order):
    """
    Returns names from all categories concatenated. There can be repeated names.
    var_names: Dictionary with tuples of names for each category
    merge_order: Iterable with the correct order in which each category should be concatenated.
    """
    concatenated_names = []
    for category in merge_order:
        concatenated_names = concatenated_names + var_names[category]
    return concatenated_names

def my_generator(file_name, set_name, batch_size=1):
    """
    Yields a batch of samples ready to use for predictions with a Keras model.
    It takes care of getting the correct variables accross datasets, preprocessing, scaling and centering.
    file_name: Path to the hdf5 file 
    set_name: Name of the set to use, for example 'hl_tracks'
    batch_size: Size of each batch to yield
    """
    var_names, merge_order = get_variable_names(set_name)
    data_file = h5py.File(file_name, 'r')
    assert data_file is not None
    total_num_samples = get_num_samples(data_file)
    mean_vector, std_vector = load_mean_and_std(set_name)
    set_variable_names = concatenate_names_from_categories(var_names, merge_order)  # in case we want to look at the full var name list

    while True:
        for start, end in zip(range(0, total_num_samples, batch_size), range(batch_size, total_num_samples+batch_size, batch_size)):
            merge_list = []
            for category in merge_order:
                # The batch of interest has variables in several of the datasets of the hdf5 file. 
                # We have to extract each one separately and then merge them. 
                # The merge also needs to be done in a specific order so it matches the ordering used for scaling and training.
                category_data = data_file.get(category)
                assert category_data is not None
                category_batch = category_data[start:end] # this may need to be adjusted depending on dimensions
                category_batch = extract_variables(category_batch, var_names[category])
                category_batch = flatten(category_batch)
                merge_list.append(category_batch)
            data_batch = merge_batches_from_categories(merge_list)
            data_batch = np.nan_to_num(data_batch)
            data_batch = scale_and_center(data_batch, mean_vector, std_vector)
            yield data_batch

if __name__ == "__main__":
    gen_1 = my_generator('small_test_raw_data_signal.h5', 'hl_tracks')
    #print(gen_1.next()) # this syntax is for python 2
    print(next(gen_1))
