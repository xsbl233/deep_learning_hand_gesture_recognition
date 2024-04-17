import glob
import numpy
import pickle
from scipy import ndimage as ndimage
from sklearn.model_selection import train_test_split


def resize_gestures(input_gestures, final_length=100):
    """
    Resize the time series by interpolating them to the same length

    Input:
        - input_gestures: list of numpy.ndarray tensors.
              Each tensor represents a single gesture.
              Gestures can have variable durations.
              Each tensor has a shape: (duration, channels)
              where duration is the duration of the individual gesture
                    channels = 44 = 2 * 22 if recorded in 2D and
                    channels = 66 = 3 * 22 if recorded in 3D 
    Output:
        - output_gestures: one numpy.ndarray tensor.
              The output tensor has a shape: (records, final_length, channels)
              where records = len(input_gestures)
                   final_length is the common duration of all gestures
                   channels is the same as above 
    """
    # please use python3. if you still use python2, important note: redefine the classic division operator / by importing it from the __future__ module
    output_gestures = numpy.array([numpy.array(
        [ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in range(numpy.size(x_i, 1))]).T for x_i
                                   in input_gestures])
    return output_gestures


def load_gestures(dataset='dhg', root='./dataset_dhg1428', version_x='3D', version_y='both',
                  resize_gesture_to_length=100):
    """
    Get the 3D or 2D pose gestures sequences, and their associated labels.

    Ouput:
        - a tuple of (gestures, labels) or (gestures, labels_14, labels_28)
              where gestures is either a numpy.ndarray tensor or
                                       a list of numpy.ndarray tensors,
                                       depending on if the gestures have been resized or not.
              Each tensor represents a single gesture.
              Gestures can have variable durations.
              Each tensor has a shape: (duration, channels) where channels is either 44 (= 2 * 22) or 66 (=3 * 22)
    """

    # SHREC 2017 (on Google Colab):
    # root = '/content/dataset_shrec2017/HandGestureDataset_SHREC2017'
    # DHG 14/28 (on Google Colab):
    # root = '/content/dataset_dhg1428'
    if dataset == 'dhg':
        assert 'dataset_dhg' in root
    if dataset == 'shrec':
        assert 'dataset_shrec' in root

    if version_x == '3D':
        if dataset == 'dhg':
            pattern = root + '/gesture_*/finger_*/subject_*/essai_*/skeleton_world.txt'
        elif dataset == 'shrec':
            pattern = root + '/gesture_*/finger_*/subject_*/essai_*/skeletons_world.txt'
    else:
        if dataset == 'dhg':
            pattern = root + '/gesture_*/finger_*/subject_*/essai_*/skeleton_image.txt'
        elif dataset == 'shrec':
            pattern = root + '/gesture_*/finger_*/subject_*/essai_*/skeletons_image.txt'

    gestures_filenames = sorted(glob.glob(pattern))
    gestures = [numpy.genfromtxt(f) for f in gestures_filenames]
    if resize_gesture_to_length is not None:
        gestures = resize_gestures(gestures, final_length=resize_gesture_to_length)

    print(gestures[0][0])
    exit(0)
    print(gestures_filenames)
    for filename in gestures_filenames:
        print(filename.split('\\'))
    labels_14 = [int(filename.split('\\')[-5].split('_')[1]) for filename in gestures_filenames]
    labels_28 = [int(filename.split('\\')[-4].split('_')[1]) for filename in gestures_filenames]
    labels_28 = [labels_14[idx_gesture] if n_fingers_used == 1 else 14 + labels_14[idx_gesture] for
                 idx_gesture, n_fingers_used in enumerate(labels_28)]
    if version_y == '14' or version_y == 14:
        return gestures, labels_14
    elif version_y == '28' or version_y == 28:
        return gestures, labels_28
    elif version_y == 'both':
        return gestures, labels_14, labels_28


def write_data(data, filepath):
    """Save the dataset to a file. Note: data is a dict with keys 'x_train', ..."""
    with open(filepath, 'wb') as output_file:
        pickle.dump(data, output_file)


def load_data(filepath='./shrec_data.pckl'):
    """
    Returns hand gesture sequences (X) and their associated labels (Y).
    Each sequence has two different labels.
    The first label  Y describes the gesture class out of 14 possible gestures (e.g. swiping your hand to the right).
    The second label Y describes the gesture class out of 28 possible gestures (e.g. swiping your hand to the right with your index pointed, or not pointed).
    """
    file = open(filepath, 'rb')
    data = pickle.load(file, encoding='latin1')  # <<---- change to 'latin1' to 'utf8' if the data does not load
    file.close()
    return data['x_train'], data['x_test'], data['y_train_14'], data['y_train_28'], data['y_test_14'], data['y_test_28']

# ---------------------------------------------------------
# Step 3. Save the dataset(s) you need
# ---------------------------------------------------------
# Example: 3D version of the SHREC17 and DHG gesture datasets, with gestures resized to 100 timesteps
gestures, labels_14, labels_28 = load_gestures(dataset='dhg',
                                               root='.\\dataset_dhg1428',
                                               version_x='3D',
                                               version_y='both',
                                               resize_gesture_to_length=100)
# Split the dataset into train and test sets if you want:
x_train, x_test, y_train_14, y_test_14, y_train_28, y_test_28 = train_test_split(gestures, labels_14, labels_28, test_size=0.20)

# Save the dataset
data = {
    'x_train': x_train,
    'x_test': x_test,
    'y_train_14': y_train_14,
    'y_train_28': y_train_28,
    'y_test_14': y_test_14,
    'y_test_28': y_test_28
}
write_data(data, filepath='dhg_data.pckl')

# ---------------------------------------------------------
# Step 5. Use the dataset(s)
# ---------------------------------------------------------
x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = load_data('dhg_data.pckl')
