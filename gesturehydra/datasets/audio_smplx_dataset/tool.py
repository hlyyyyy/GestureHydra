"""
This script contains supporting function for the data processing.
It is used in several other scripts:
for generating bvh files, aligning sequences and calculation of speech features

@author: Taras Kucherenko
"""
from pydub import AudioSegment
import parselmouth as pm
import os
import numpy as np


def derivative(x, f):
    """ Calculate numerical derivative (by FDM) of a 1d array
    Args:
        x: input space x
        f: Function of x
    Returns:
        der:  numerical derivative of f wrt x
    """

    x = 1000 * x  # from seconds to milliseconds

    # Normalization:
    dx = (x[1] - x[0])

    cf = np.convolve(f, [1, -1]) / dx

    # Remove unstable values
    der = cf[:-1].copy()
    der[0] = 0

    return der


def create_bvh(filename, prediction, frame_time):
    """
    Create BVH File
    Args:
        filename:    file, in which motion in bvh format should be written
        prediction:  motion sequences, to be written into file
        frame_time:  frame rate of the motion
    Returns:
        nothing, writes motion to the file
    """
    with open('hformat.txt', 'r') as ftemp:
        hformat = ftemp.readlines()

    with open(filename, 'w') as fo:
        prediction = np.squeeze(prediction)
        print("output vector shape: " + str(prediction.shape))
        offset = [0, 60, 0]
        offset_line = "\tOFFSET " + " ".join("{:.6f}".format(x) for x in offset) + '\n'
        fo.write("HIERARCHY\n")
        fo.write("ROOT Hips\n")
        fo.write("{\n")
        fo.write(offset_line)
        fo.writelines(hformat)
        fo.write("MOTION\n")
        fo.write("Frames: " + str(len(prediction)) + '\n')
        fo.write("Frame Time: " + frame_time + "\n")
        for row in prediction:
            row[0:3] = 0
            legs = np.zeros(24)
            row = np.concatenate((row, legs))
            label_line = " ".join("{:.6f}".format(x) for x in row) + " "
            fo.write(label_line + '\n')
        print("bvh generated")


def shorten(arr1, arr2, min_len=0):

    if min_len == 0:
        min_len = min(len(arr1), len(arr2))

    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]

    return arr1, arr2


def average(arr, n):
    """ Replace every "n" values by their average
    Args:
        arr: input array
        n:   number of elements to average on
    Returns:
        resulting array
    """
    end = n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def extract_prosodic_features(audio_filename, FPS):
    """
    Extract all 5 prosodic features
    Args:
        audio_filename:   file name for the audio to be used
    Returns:
        pros_feature:     energy, energy_der, pitch, pitch_der, pitch_ind
    """
    HOP_LENGTH = 1/FPS
    # Read audio from file
    sound = AudioSegment.from_file(audio_filename, format="wav")

    # Alternative prosodic features
    pitch, energy = compute_prosody(audio_filename, HOP_LENGTH / 10)

    duration = len(sound) / 1000
    t = np.arange(0, duration, HOP_LENGTH / 10)

    energy_der = derivative(t, energy)
    pitch_der = derivative(t, pitch)

    # Average everything in order to match the frequency
    energy = average(energy, 10)
    energy_der = average(energy_der, 10)
    pitch = average(pitch, 10)
    pitch_der = average(pitch_der, 10)

    # Cut them to the same size
    min_size = min(len(energy), len(energy_der), len(pitch_der), len(pitch_der))
    energy = energy[:min_size]
    energy_der = energy_der[:min_size]
    pitch = pitch[:min_size]
    pitch_der = pitch_der[:min_size]

    # Stack them all together
    pros_feature = np.stack((energy, energy_der, pitch, pitch_der))#, pitch_ind))

    # And reshape
    pros_feature = np.transpose(pros_feature)

    return pros_feature


def compute_prosody(audio_filename, time_step=0.05):
    audio = pm.Sound(audio_filename)

    # Extract pitch and intensity
    pitch = audio.to_pitch(time_step=time_step)
    intensity = audio.to_intensity(time_step=time_step)

    # Evenly spaced time steps
    times = np.arange(0, audio.get_total_duration() - time_step, time_step)

    # Compute prosodic features at each time step
    pitch_values = np.nan_to_num(
        np.asarray([pitch.get_value_at_time(t) for t in times]))
    intensity_values = np.nan_to_num(
        np.asarray([intensity.get_value(t) for t in times]))

    intensity_values = np.clip(
        intensity_values, np.finfo(intensity_values.dtype).eps, None)

    # Normalize features [Chiu '11]
    pitch_norm = np.clip(np.log(pitch_values + 1) - 4, 0, None)
    intensity_norm = np.clip(np.log(intensity_values) - 3, 0, None)

    return pitch_norm, intensity_norm