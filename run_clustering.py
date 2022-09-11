"""
This script runs a clustering algorithm on Placenta MRI data.
The clustering algorithm is in SKlearn convention (e.g fit and predict functions)

Created by Elisha Goldstein Aug 2021
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from PIL import Image
from fcmeans import FCM
from scipy.stats import mode
from argparse import ArgumentParser
from scipy.ndimage import generic_filter
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from general_utils import save_how_to


def get_modal(arr):
    return mode(arr, axis=None)[0]


def make_gif_from_img_dir(images_dir_path):
    imgs_paths = glob.glob(images_dir_path + '/*.png')
    print("number of png images found:", len(imgs_paths))

    imgs = [Image.open(images_dir_path + '/mask{}.png'.format(i)) for i in range(len(imgs_paths))]

    # create GIF
    imgs[0].save(os.path.join(images_dir_path + 'mask_gif.gif'), format='GIF',
                 append_images=imgs[1:], save_all=True, duration=400, loop=0)


def prepare_data(cropped_matrices, all_samples_names: list, sname: str):
    if sname == '4_control':
        data_matrix = np.stack(cropped_matrices[:4])
        all_samples_names = all_samples_names[:4]
    elif sname == '4_diabetes':
        data_matrix = np.stack(cropped_matrices[4:])
        all_samples_names = all_samples_names[4:]
    elif sname == '13_samples':
        data_matrix = np.stack(cropped_matrices[:])
        all_samples_names = all_samples_names
    else:
        print("ERROR: sname is not good")

    return data_matrix, all_samples_names


def run(args):
    sname = '13_samples'  # {4_control, 4_diabetes, 8_samples}
    algo = args.algo + '_'  # {fcm_, bgm_, ggm_}
    list_of_k = [int(k) for k in args.k]

    # load the raw data martix
    data_matrix = np.load(args.input_data)
    # cropped_matrices = np.load(
    #     '/home/labs/neeman/Collaboration/Placenta_MRI_2021/data/raw_data_dropbox/4channels/13_samples_4ch_digbiopsy.npy')
    # cropped_matrices = np.load('/home/labs/bioservices/eligol/01_projects/06_pregnancy_MRI/data/8_samp_cropped_norm.npy')

    all_samples_names = ['emc_0117', 'emc_0201', 'mmc_0122', 'hmo_0252', 'mmc_0178', 'hymc_0085', 'mmc_0255',
                         'mmc_0084',
                         'mmc_0129', 'mmc_0123', 'mmc_0086', 'mmc_0228', 'mmc_0128']

    # data_matrix = data_matrix[:, :5, :5, :5, :]

    data_matrix, all_samples_names = prepare_data(data_matrix, all_samples_names, sname)

    # reshape the 3d array to a vector
    print("shpae before reshape:", data_matrix.shape)
    vec_to_cluster = np.reshape(data_matrix, newshape=(
        data_matrix.shape[0] * data_matrix.shape[1] * data_matrix.shape[2] * data_matrix.shape[3],
        data_matrix.shape[4]))
    print("shpae after reshape:", vec_to_cluster.shape)

    # make clustering
    for k in list_of_k:
        print("K is", k)

        if algo == 'fcm_':
            model_name = 'FCM'
            model = FCM(n_clusters=k)
        elif algo == 'bgm_':
            model_name = 'BGM'
            model = BayesianGaussianMixture(n_components=k)
        elif algo == 'ggm_':
            model_name = 'GGM'
            model = GaussianMixture(n_components=k)

        # create directory
        out_dir = args.output_dir
        path_to_save_files = os.path.join(out_dir, '{}_4channels_labels/{}_{}_means'.format(model_name, sname, k))
        os.makedirs(path_to_save_files, exist_ok=True)
        print('saving files to ', path_to_save_files)

        # fit the model
        model.fit(vec_to_cluster)

        # make preditction
        labels_mat = model.predict(vec_to_cluster)

        # reshape the predictions vector back to 3D array
        # FIXME: try selfexplainable code, and reserve in-line comments
        data_labeled = np.reshape(labels_mat, newshape=(
            data_matrix.shape[0], data_matrix.shape[1], data_matrix.shape[2], data_matrix.shape[3]))
        all_labels_discrete = np.array(data_labeled, dtype='int8')  # convert to 'int8'. spare memory

        # save the labels matrix
        np.save(os.path.join(path_to_save_files, 'labels_mask_4ch.npy'), all_labels_discrete)

        # save the model
        with open(os.path.join(path_to_save_files, "{}_K{}_{}_model.pkl".format(model_name, k, sname)), "wb") as f:
            pickle.dump(model, f)

        # save the centers to text file
        try:
            np.savetxt(os.path.join(path_to_save_files, '{}_K{}_{}_centers.txt'.format(model_name, k, sname)),
                       model.centers)
        except AttributeError as e:
            print("model doesn't have centers to save. Load the model from pickle and look for other attributes")

        # save the all_labels_discrete in separate directories
        for sample_ids, sample_name in enumerate(all_samples_names):
            print("Sample:", sample_name)
            # make separate labels
            for label in np.unique(all_labels_discrete):

                # create a direcoty
                dir_path = os.path.join(path_to_save_files, sample_name, 'no_filter', 'label_{}'.format(label))
                os.makedirs(dir_path, exist_ok=True)

                # get discrte labels
                label_x = np.array((all_labels_discrete == label) + 0, dtype='int8')

                # save the masks in slices
                for j in range(label_x.shape[1]):
                    plt.imsave(os.path.join(dir_path, '{}_mask{}.tiff'.format(sname, j)), label_x[sample_ids, j, :, :],
                               cmap='tab20')

            # save slices with all labels with filter
            print("starting to make labels with filter")
            path = os.path.join(path_to_save_files, sample_name, 'filter33')
            os.makedirs(path, exist_ok=True)

            for i in range(all_labels_discrete.shape[1]):
                im = generic_filter(all_labels_discrete[sample_ids, i, :, :], get_modal, (3, 3))
                plt.imsave(os.path.join(path, 'mask{}.png'.format(i)), im, cmap='tab20')

            # make gif
            print("starting to make gif")
            make_gif_from_img_dir(path)


########################################################################################################################

def fill_parser(parser):
    parser.add_argument('input_data', help='path to data matrix')

    parser.add_argument('output_dir', help='Output folder to save the CV training')

    parser.add_argument('--k', nargs='+', default=[3], help="list of Ks to make clustering on")

    parser.add_argument('--algo', type=str, default='fcm', help='clustering algorithm from skleran')

    return parser


def get_args():
    parser = ArgumentParser()
    fill_parser(parser)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run(args)

    try:
        save_how_to(args.output_dir, sub_cmd_str='CV_train')
    except Exception as e:
        print(e)
        print('If the above error is: "Unknown option: -C", you need to use a newer git version. try "module load git"')

    print("\nfinished")
