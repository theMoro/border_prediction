"""
Author: Tobias Morocutti
Matr.Nr.: K12008172
Exercise 5
"""

import glob
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch.nn.functional as F

import utils
import Net
from supplements_ex5 import scoring

import matplotlib.pyplot as plt

SEED_NR = 43
OUTPUT_PATH = '../my_output_files/'
PREDICTIONS_PATH = OUTPUT_PATH + 'my_predictions.pkl'
TRAINSET_PATH = OUTPUT_PATH + 'trainset.h5'
VALSET_PATH = '../supplements_ex5/example_testset.pkl'
VALSET_TARGETS_PATH = '../supplements_ex5/example_targets.pkl'
TESTSET_PATH = '../challenge_testset/testset.pkl'

BATCH_SIZE = Net.BATCH_SIZE
device = Net.device

def get_input_known_and_target_arrays_ex4(image_array: np.ndarray, border_x: tuple, border_y: tuple):
    """See assignment sheet for usage description"""
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        raise NotImplementedError("image_array must be a 2D numpy array")

    border_x_start, border_x_end = border_x
    border_y_start, border_y_end = border_y

    try:  # Check for conversion to int (would raise ValueError anyway but we will write a nice error message)
        border_x_start = int(border_x_start)
        border_x_end = int(border_x_end)
        border_y_start = int(border_y_start)
        border_y_end = int(border_y_end)
    except ValueError as e:
        raise ValueError(f"Could not convert entries in border_x and border_y ({border_x} and {border_y}) to int! "
                         f"Error: {e}")

    if border_x_start < 1 or border_x_end < 1:
        raise ValueError(f"Values of border_x must be greater than 0 but are {border_x_start, border_x_end}")

    if border_y_start < 1 or border_y_end < 1:
        raise ValueError(f"Values of border_y must be greater than 0 but are {border_y_start, border_y_end}")

    remaining_size_x = image_array.shape[0] - (border_x_start + border_x_end)
    remaining_size_y = image_array.shape[1] - (border_y_start + border_y_end)
    if remaining_size_x < 16 or remaining_size_y < 16:
        raise ValueError(f"the size of the remaining image after removing the border must be greater equal (16,16) "
                         f"but was ({remaining_size_x},{remaining_size_y})")

    # Create known_array
    known_array = np.zeros_like(image_array)
    known_array[border_x_start:-border_x_end, border_y_start:-border_y_end] = 1

    # Create target_array - don't forget to use .copy(), otherwise target_array and image_array might point to the
    # same array!
    target_array = image_array[known_array == 0].copy()

    mean, std = np.mean(image_array[known_array == 1]), np.std(image_array[known_array == 1])

    # Use image_array as input_array
    image_array[known_array == 0] = ((127.5 / 255 * std) + mean) * 255 # 1 / 2

    return image_array, known_array, target_array, mean, std


def add_normal_noise(input_tensor, mean, std):
    # Create the tensor containing the noise
    noise_tensor = torch.empty_like(input_tensor)
    noise_tensor.normal_(mean=mean, std=std)
    # Add noise to input tensor and return results
    return input_tensor + noise_tensor


def wrap_add_normal_noise(mean: float = 0.48, std: float = 0.19):
    def noisy_image(input_tensor):
        input_tensor = add_normal_noise(input_tensor, mean, std)
        return input_tensor

    return noisy_image


noise_transform = transforms.Lambda(lambd=wrap_add_normal_noise())


class ImageTransform():
    def __init__(self, mean=0, std=0, im_shape=(90, 90)):

        self.data_transform = {
            'normalize':
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),
            'resize':
                transforms.Compose([
                    transforms.Resize(size=im_shape[0]),
                    transforms.CenterCrop(size=(im_shape))
                ]),
            'train':
                transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    noise_transform,
                    transforms.RandomErasing()
                ]),
            'val':
                transforms.Compose([
                    transforms.Resize(size=im_shape[0]),
                    transforms.CenterCrop(size=(im_shape)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),
            'test':
                transforms.Compose([
                    transforms.Resize(size=im_shape[0]),
                    transforms.CenterCrop(size=(im_shape)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


class ImageDataset(Dataset):
    def __init__(self, input_arrays, known_arrays, borders_x, borders_y, sample_ids, target_arrays,
                 means=None, stds=None, phase=None):
        self.input_arrays = input_arrays
        self.known_arrays = known_arrays
        self.borders_x = borders_x
        self.borders_y = borders_y
        self.sample_ids = sample_ids
        self.target_arrays = target_arrays
        self.means = means
        self.stds = stds
        self.phase = phase

        self.transform = ImageTransform()

    def __getitem__(self, idx):

        input_arr = np.array(self.input_arrays[idx], dtype=np.uint8)
        mask = np.array(self.known_arrays[idx], dtype=np.uint8)
        mask_img = mask.copy()
        mask_img[mask == 0] = 255
        mask_img[mask == 1] = 0

        if self.phase != 'test':
            target_arr = np.array(self.target_arrays[idx], dtype=np.uint8)  # should stay denormalized
            target_a = input_arr.copy()
            target_arr_trimmed = np.trim_zeros(target_arr)

            if len(target_arr_trimmed) < len(target_a[mask == 0]):
                target_arr_trimmed = list(target_arr_trimmed)
                target_arr_trimmed.extend([0] * (len(target_a[mask == 0]) - len(target_arr_trimmed)))
                target_arr_trimmed = np.array(target_arr_trimmed)

            target_a[mask == 0] = target_arr_trimmed
            target_arr = target_a

        sample_id = int(self.sample_ids[idx])

        if self.phase in ['val', 'test']:
            if self.phase == 'val':
                target_img = Image.fromarray(target_arr)
                target_img = self.transform(target_img, 'resize')
                target_arr = np.array(target_img, dtype=np.uint8)

            mean, std = np.mean(input_arr[mask == 1]) / 255, np.std(input_arr[mask == 1]) / 255

            input_arr[mask == 0] = ((127.5 / 255 * std) + mean) * 255 # 1 / 2
            input_arr = Image.fromarray(input_arr)
            mask_img = Image.fromarray(mask_img)
        else:
            mean, std = self.means[idx] / 255, self.stds[idx] / 255


        self.transform = ImageTransform(mean=mean, std=std)
        # print("phaseee: ", self.phase)
        normalized_input_tensor = self.transform(input_arr, 'normalize') # self.phase
        normalized_known_tensor = self.transform(mask_img, 'normalize')

        if self.phase == 'test':
            return normalized_input_tensor, normalized_known_tensor, None, torch.tensor(mask), \
                   mean, std, sample_id
        else:
            return normalized_input_tensor, normalized_known_tensor, torch.tensor(target_arr), torch.tensor(mask),\
                   mean, std, sample_id

    def __len__(self):
        return len(self.sample_ids)


def stack_collate_fn_general(batch_as_list: list, phase):
    norm_input_arrays = [sample[0] for sample in batch_as_list]
    norm_known_arrays = [sample[1] for sample in batch_as_list]

    stacked_input_tensor = torch.nn.utils.rnn.pad_sequence(
        [torch.stack(norm_input_arrays), torch.stack(norm_known_arrays)],
        padding_value=0).view(-1, 2, 90, 90)

    if phase == 'train':
        unnorm_target_arrays = [sample[2] for sample in batch_as_list]
        unnorm_target_tensor = torch.stack(unnorm_target_arrays)
    else:
        unnorm_target_tensor = None

    unnorm_mask_arrays = [sample[3] for sample in batch_as_list]
    unnorm_mask_tensor = torch.stack(unnorm_mask_arrays)

    means = [sample[4] for sample in batch_as_list]
    means_tensor = torch.tensor(means)

    stds = [sample[5] for sample in batch_as_list]
    stds_tensor = torch.tensor(stds)

    sample_ids = [sample[6] for sample in batch_as_list]
    sample_ids_tensor = torch.tensor(sample_ids)

    return stacked_input_tensor, unnorm_target_tensor, unnorm_mask_tensor, \
           means_tensor, stds_tensor, sample_ids_tensor


def stack_collate_fn(batch_as_list: list):
    return stack_collate_fn_general(batch_as_list, phase='train')


def stack_collate_fn_test(batch_as_list: list):
    return stack_collate_fn_general(batch_as_list, phase='test')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_images_and_write_to_file():
    tqdm.write("reading images and saving them to file ...")

    img_paths = glob.glob(os.path.join('../dataset', '**', '*.jpg'), recursive=True)

    all_inputs, all_knowns, all_targets, all_sample_ids = [], [], [], []
    all_means, all_stds = [], []
    random_borders_1 = np.random.randint(5, 10, len(img_paths))
    random_borders_2 = [random.randint(5, 14 - nr) for nr in random_borders_1]

    borders_x = np.array([(random_borders_1[i], random_borders_2[i]) for i in range(len(img_paths))], dtype=np.uint8)
    borders_y = borders_x.copy()

    output_borders_x, output_borders_y = borders_x[:500], borders_y[:500]

    transform = ImageTransform((90, 90))

    for idx, path in enumerate(tqdm(img_paths)):
        img = Image.open(path)
        img = transform(img, 'resize')
        img_array = np.array(img)

        input_array, known_array, target_array, mean, std = get_input_known_and_target_arrays_ex4(
            img_array, borders_x[idx], borders_y[idx])

        all_inputs.append(input_array)
        all_knowns.append(known_array)
        all_targets.append(target_array)
        all_sample_ids.append(str(idx))
        all_means.append(mean)
        all_stds.append(std)

        if ((idx+1) % 500 == 0 and idx != 0) or idx + 1 == len(img_paths):

            all_inputs = np.array(all_inputs, dtype=np.uint8)
            all_knowns = np.array(all_knowns, dtype=np.uint8)
            all_targets = torch.nn.utils.rnn.pad_sequence(
                [torch.from_numpy(x) for x in all_targets], batch_first=True,
                padding_value=0).numpy()

            target_max_length = 2475
            target_template = np.full(shape=(all_targets.shape[0], target_max_length), fill_value=0)
            target_template[:all_targets.shape[0], :all_targets.shape[1]] = all_targets
            all_targets = np.array(target_template, dtype=np.uint8)

            all_sample_ids = np.array(all_sample_ids, dtype=np.uint8)

            all_means = np.array(all_means)
            all_stds = np.array(all_stds)

            with h5py.File(TRAINSET_PATH, 'a') as f:
                if (idx+1) == 500 or (idx + 1 == len(img_paths) and idx < 500):

                    nr_of_new_entries = len(all_inputs)
                    output_borders_x = borders_x[idx+1-nr_of_new_entries:idx+1]
                    output_borders_y = borders_y[idx+1-nr_of_new_entries:idx+1]

                    f.create_dataset('input_arrays', data=all_inputs, compression='gzip', chunks=True,
                                     maxshape=(None, 90, 90))
                    f.create_dataset('known_arrays', data=all_knowns, compression='gzip', chunks=True,
                                     maxshape=(None, 90, 90))
                    f.create_dataset('borders_x', data=output_borders_x, compression='gzip', chunks=True,
                                     maxshape=(None, 2))
                    f.create_dataset('borders_y', data=output_borders_y, compression='gzip', chunks=True,
                                     maxshape=(None, 2))
                    f.create_dataset('sample_ids', data=all_sample_ids, compression='gzip', chunks=True,
                                     maxshape=(None,))
                    f.create_dataset('targets', data=all_targets, compression='gzip', chunks=True,
                                     maxshape=(None, 2475))
                    f.create_dataset('means', data=all_means, compression='gzip', chunks=True,
                                     maxshape=(None,))
                    f.create_dataset('stds', data=all_stds, compression='gzip', chunks=True,
                                     maxshape=(None,))
                else:
                    nr_of_new_entries = 500
                    if (idx+1) % 500 != 0 and idx != 0 and idx+1 == len(img_paths):
                        nr_of_new_entries = (idx % 500) + 1

                    output_borders_x = borders_x[idx+1-nr_of_new_entries:idx+1]
                    output_borders_y = borders_y[idx+1-nr_of_new_entries:idx+1]

                    f['input_arrays'].resize(f['input_arrays'].shape[0] + nr_of_new_entries, axis=0)
                    f['input_arrays'][-nr_of_new_entries:] = all_inputs

                    f['known_arrays'].resize(f['known_arrays'].shape[0] + nr_of_new_entries, axis=0)
                    f['known_arrays'][-nr_of_new_entries:] = all_knowns

                    f['borders_x'].resize(f['borders_x'].shape[0] + nr_of_new_entries, axis=0)
                    f['borders_x'][-nr_of_new_entries:] = output_borders_x

                    f['borders_y'].resize(f['borders_y'].shape[0] + nr_of_new_entries, axis=0)
                    f['borders_y'][-nr_of_new_entries:] = output_borders_y

                    f['sample_ids'].resize(f['sample_ids'].shape[0] + nr_of_new_entries, axis=0)
                    f['sample_ids'][-nr_of_new_entries:] = all_sample_ids

                    f['targets'].resize(f['targets'].shape[0] + nr_of_new_entries, axis=0)
                    f['targets'][-nr_of_new_entries:] = all_targets

                    f['means'].resize(f['means'].shape[0] + nr_of_new_entries, axis=0)
                    f['means'][-nr_of_new_entries:] = all_means

                    f['stds'].resize(f['stds'].shape[0] + nr_of_new_entries, axis=0)
                    f['stds'][-nr_of_new_entries:] = all_stds

            data = dict()
            all_inputs, all_knowns, all_targets, all_sample_ids = [], [], [], []
            all_means, all_stds = [], []
            output_borders_x, output_borders_y = [], []


def read_data():
    set_seed(SEED_NR)

    f = h5py.File(TRAINSET_PATH, 'r')

    input_arrays = f['input_arrays']
    known_arrays = f['known_arrays']
    borders_x = f['borders_x']
    borders_y = f['borders_y']
    sample_ids = f['sample_ids']
    target_arrays = f['targets']
    means = f['means']
    stds = f['stds']

    image_dataset = ImageDataset(input_arrays, known_arrays, borders_x, borders_y, sample_ids, target_arrays,
                                 means=means, stds=stds, phase='train')

    # read val data
    val_set_data = utils.read_pickle_file(VALSET_PATH)
    val_set_targets = utils.read_pickle_file(VALSET_TARGETS_PATH)

    val_dataset = ImageDataset(val_set_data['input_arrays'], val_set_data['known_arrays'],
                                val_set_data['borders_x'], val_set_data['borders_y'],
                                [str(x) for x in np.arange(len(val_set_data['input_arrays']))],
                               val_set_targets, means=None, stds=None, phase='val')

    train_loader = DataLoader(image_dataset,
                              shuffle=True,
                              batch_size=BATCH_SIZE,
                              num_workers=8,
                              collate_fn=stack_collate_fn
                              )

    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=BATCH_SIZE,
                            num_workers=0,
                            collate_fn=stack_collate_fn
                            )

    return train_loader, val_loader


def visualize_predictions():
    preds = utils.read_pickle_file(PREDICTIONS_PATH)
    targets = utils.read_pickle_file(VALSET_TARGETS_PATH)
    testset_data = utils.read_pickle_file(VALSET_PATH)
    knowns = testset_data['known_arrays']
    inputs = testset_data['input_arrays']

    for i, pred in enumerate(preds):
        input_image = Image.fromarray(inputs[i])

        known_image = Image.fromarray(knowns[i])

        target_arr = np.zeros(knowns[i].shape)
        target_arr[knowns[i] == 0] = targets[i]
        target_image = Image.fromarray(target_arr)

        pred_arr = np.zeros(knowns[i].shape)
        pred_arr[knowns[i] == 0] = pred
        pred_image = Image.fromarray(pred_arr)

        utils.visualize_images(input_image, known_image, target_image, pred_image)


if __name__ == '__main__':
    set_seed(SEED_NR)

    if not os.path.isfile(TRAINSET_PATH):
        read_images_and_write_to_file()

    train_loader, val_loader = read_data()

    try:
        if os.path.isfile('results/models/trained_model_stopped_early.pt'):
            model = Net.Net()
            model = (torch.load('results/models/trained_model_stopped_early.pt'))
        else:
            model = Net.Net().to(device)
            Net.train(model, train_loader, val_loader, torch.nn.MSELoss(reduction="mean").to(device))

    except Exception as e:
        print(e)
        print(e)
        print("---")
    finally:
        loss_train, train_preds = Net.evaluate_model(model, train_loader, torch.nn.MSELoss(reduction="mean"), phase='train')
        print('')
        print('Training set: ')
        print('Loss: %g' % (loss_train))
        print('')

        loss_val, val_predictions = Net.evaluate_model(model, val_loader, torch.nn.MSELoss(reduction="mean"), phase='val')
        print('')
        print('Validation set: ')
        print('Loss: %g' % (loss_val))
        print('')

        utils.write_dic_to_pickle_file(val_predictions, PREDICTIONS_PATH)
        score = scoring.scoring(PREDICTIONS_PATH, VALSET_TARGETS_PATH)
        print("example test score: %g" % score)

        # official test set
        testset_data = utils.read_pickle_file(TESTSET_PATH)

        test_dataset = ImageDataset(testset_data['input_arrays'], testset_data['known_arrays'],
                                    testset_data['borders_x'], testset_data['borders_y'],
                                    [str(x) for x in np.arange(len(testset_data['input_arrays']))],
                                    None, means=None, stds=None, phase='test')

        test_loader = DataLoader(test_dataset,
                                 shuffle=False,
                                 batch_size=BATCH_SIZE,
                                 num_workers=0,
                                 collate_fn=stack_collate_fn_test
                                 )

        _, test_predictions = Net.evaluate_model(model, test_loader, torch.nn.MSELoss(reduction="mean"), phase="test")

        utils.write_dic_to_pickle_file(test_predictions, PREDICTIONS_PATH)

