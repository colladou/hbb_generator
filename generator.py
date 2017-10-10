from __future__ import print_function

import numpy as np
import h5py
from os.path import join

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
    mean_vector = np.load(join(load_path, "%s_mean_vector.npy" % set_name))
    std_vector = np.load(join(load_path, "%s_std_vector.npy" % set_name))
    std_vector[std_vector == 0] = 1  # prevent x/0 division
    assert np.sum(np.isnan(mean_vector)) == 0, "Nan value found in mean vector"
    assert np.sum(np.isnan(std_vector)) == 0, "Nan value found in std vector"
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
    print(data.shape)
    print(data)
    print(data[variable_names])
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
            data_batch = np.hstack((data_batch, category_batch))
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
        concatenated_names = concatenated_names + list(var_names[category])
    return concatenated_names

def get_weights(data_file, start=0, end=None):
    """
    Returns the weights of the data. It assumes they are in the jets dataset
    data_file: Open file with the data
    start: batch start sample
    end: batch end sample
    """
    category_data = data_file.get('jets')
    if end is None:
        end = category_data.shape[0]
    category_batch = category_data[start:end]
    weights = category_batch['weight']
    assert weights is not None
    return weights

def get_batch_slicing_indexes(batch_size, total_num_samples):
    """
    Returns an iterable with the (start, end) indices for slicing the data into batches.
    It solves the problem of the number of samples not being divisible by the batch size
    batch_size: Desired batch size all the batches except the last one in case it is not divisible
    total_num_samples: Total number of samples in the dataset
    """
    num_trailing_samples = total_num_samples%batch_size
    num_samples = total_num_samples - num_trailing_samples
    if num_trailing_samples != 0:
        batch_start_end_indices = zip(list(range(0, num_samples, batch_size)) + [num_samples], 
                                      list(range(batch_size, num_samples+batch_size, batch_size)) + [total_num_samples])
    else:
        batch_start_end_indices = zip(range(0, num_samples, batch_size),
                                      range(batch_size, num_samples+batch_size, batch_size))
    return batch_start_end_indices

def my_generator(file_name, set_name, batch_size=1, label=None, include_weights=False, mean_and_std_path='models'):
    """
    Yields a batch of samples ready to use for predictions with a Keras model.
    It takes care of getting the correct variables accross datasets, preprocessing, scaling and centering.
    file_name: Path to the hdf5 file 
    set_name: Name of the set to use, for example 'hl_tracks'
    batch_size: Size of each batch to yield
    label: Integer, specifies the label to be returned for this sample
    """
    var_names, merge_order = get_variable_names(set_name)
    data_file = h5py.File(file_name, 'r')
    assert data_file is not None
    total_num_samples = get_num_samples(data_file)
    mean_vector, std_vector = load_mean_and_std(set_name, mean_and_std_path)
    set_variable_names = concatenate_names_from_categories(var_names, merge_order)  # in case we want to look at the full var name list

    while True:
        for start, end in get_batch_slicing_indexes(batch_size, total_num_samples):
            merge_list = []
            for category in merge_order:
                # The batch of interest has variables in several of the datasets of the hdf5 file. 
                # We have to extract each one separately and then merge them. 
                # The merge also needs to be done in a specific order so it matches the ordering used for scaling and training.
                category_data = data_file.get(category)
                assert category_data is not None
                category_batch = category_data[start:end] # this may need to be adjusted depending on dimensions
                category_batch = category_batch[list(var_names[category])]  
                category_batch = flatten(category_batch, var_names[category])
                merge_list.append(category_batch)
            data_batch = merge_batches_from_categories(merge_list)
            data_batch = np.nan_to_num(data_batch)
            data_batch = scale_and_center(data_batch, mean_vector, std_vector)

            if include_weights:
                weights = get_weights(data_file, start, end)

            if label is not None and include_weights:
                y = np.ones((data_batch.shape[0],)) * label
                yield [data_batch, y, weights]
            elif label is None and include_weights:
                yield [data_batch, weights]
            elif label is not None and not include_weights:
                yield [data_batch, y]
            else:
                yield data_batch

if __name__ == "__main__":
    gen_1 = my_generator('small_test_raw_data_signal.h5', 'hl_tracks', 2, 1)
    #print(gen_1.next()) # this syntax is for python 2
    print(next(gen_1))
    # this tests the trailing samples problem
    #gen_2 = my_generator('small_test_raw_data_signal.h5', 'hl_tracks', 100)
    #for i in range(88):
    #    batch = next(gen_2)
    #    print(batch[0].shape)

