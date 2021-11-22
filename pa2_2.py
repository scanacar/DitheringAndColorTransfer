import cv2 as cv2
import numpy as np


#  This function computes mean and variance values of l, a, b channels of images
def compute_mean_and_variances(image):

    (l, a, b) = cv2.split(image)
    (mean_l, var_l) = (np.mean(l), np.var(l))
    (mean_a, var_a) = (np.mean(a), np.var(a))
    (mean_b, var_b) = (np.mean(b), np.var(b))

    return mean_l, var_l, mean_a, var_a, mean_b, var_b


#  Color Transfer function
def colorTransfer(source, target):

    #  For converting RGB space to LMS space
    rgb_to_lms = np.array([[0.3811, 0.5783, 0.0402],
                           [0.1967, 0.7244, 0.0782],
                           [0.0241, 0.1288, 0.8444]])

    #  For converting LMS space to RGB space
    lms_to_rgb = np.array([[4.4679, -3.5873, 0.1193],
                           [-1.2186, 2.3809, -0.1624],
                           [0.0497, -0.2439, 1.2045]])

    #  For converting LMS space to Lab space
    lms_to_lab_1 = np.array([[1, 1, 1],
                            [1, 1, -2],
                            [1, -1, 0]])

    #  For converting LMS space to Lab space
    lms_to_lab_2 = np.array([[1/np.sqrt(3), 0, 0],
                            [0, 1/np.sqrt(6), 0],
                            [0, 0, 1/np.sqrt(2)]])

    #  For converting Lab space to LMS space
    lab_to_lms_1 = np.array([[1, 1, 1],
                            [1, 1, -1],
                            [1, -2, 0]])

    #  For converting Lab space to LMS space
    lab_to_lms_2 = np.array([[np.sqrt(3)/3, 0, 0],
                            [0, np.sqrt(6)/6, 0],
                            [0, 0, np.sqrt(2)/2]])

    #  Converting to LMS space from RGB space
    r_s, g_s, b_s = source.shape
    r_t, g_t, b_t = target.shape

    reshaped_source = source.transpose(2, 0, 1).reshape(3, -1)
    reshaped_target = target.transpose(2, 0, 1).reshape(3, -1)

    lms_source = rgb_to_lms @ reshaped_source
    lms_target = rgb_to_lms @ reshaped_target

    lms_source = lms_source.reshape(b_s, r_s, g_s).transpose(1, 2, 0).astype(np.uint8)
    lms_target = lms_target.reshape(b_t, r_t, g_t).transpose(1, 2, 0).astype(np.uint8)

    """cv2.imshow("LMS Source", lms_source)
    cv2.imshow("LMS Target", lms_target)"""

    # Converting to Logarithmic Space from LMS space
    c_1 = 255 / np.log(1 + np.max(lms_source))
    c_2 = 255 / np.log(1 + np.max(lms_target))
    log_source = c_1 * (np.log(lms_source + 1))
    log_target = c_2 * (np.log(lms_target + 1))

    log_source = np.array(log_source, dtype=np.uint8)
    log_target = np.array(log_target, dtype=np.uint8)

    """cv2.imshow("LOG Source", log_source)
    cv2.imshow("LOG Target", log_target)"""

    # Converting to LAB Space from Logarithmic Space
    reshaped_log_source = log_source.transpose(2, 0, 1).reshape(3, -1)
    reshaped_log_target = log_target.transpose(2, 0, 1).reshape(3, -1)

    lab_source = lms_to_lab_2 @ lms_to_lab_1 @ reshaped_log_source
    lab_target = lms_to_lab_2 @ lms_to_lab_1 @ reshaped_log_target

    lab_source = lab_source.reshape(b_s, r_s, g_s).transpose(1, 2, 0).astype(np.uint8)
    lab_target = lab_target.reshape(b_t, r_t, g_t).transpose(1, 2, 0).astype(np.uint8)

    """cv2.imshow("LAB Source", lab_source)
    cv2.imshow("LAB Target", lab_target)"""

    #  OPENCV Function
    lab_source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    lab_target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # Computing Means and Variances of l, a, b channels
    (mean_l_src, var_l_src, mean_a_src, var_a_src, mean_b_src, var_b_src) = compute_mean_and_variances(lab_source)
    (mean_l_trg, var_l_trg, mean_a_trg, var_a_trg, mean_b_trg, var_b_trg) = compute_mean_and_variances(lab_target)

    # STEP 5
    (l, a, b) = cv2.split(lab_source)
    result_l = l - mean_l_src
    result_a = a - mean_a_src
    result_b = b - mean_b_src

    # STEP 6
    result_l = (var_l_trg / var_l_src) * result_l
    result_a = (var_a_trg / var_a_src) * result_a
    result_b = (var_b_trg / var_b_src) * result_b
    
    # STEP 7
    result_l = result_l + mean_l_trg
    result_a = result_a + mean_a_trg
    result_b = result_b + mean_b_trg

    result_l = np.clip(result_l, 0, 255)
    result_a = np.clip(result_a, 0, 255)
    result_b = np.clip(result_b, 0, 255)

    #  Applying color transfer between two images
    transfer = cv2.merge([result_l, result_a, result_b])

    #  OPENCV Function
    #  rgb_transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    #  Converting to LMS space from Lab space
    reshaped_transfer = transfer.transpose(2, 0, 1).reshape(3, -1)
    lms_transfer = lab_to_lms_1 @ lab_to_lms_2 @ reshaped_transfer
    lms_transfer = lms_transfer.reshape(b_s, r_s, g_s).transpose(1, 2, 0).astype(np.uint8)

    #  Converting to Linear space from LMS space
    (l_t, a_t, b_t) = cv2.split(lms_transfer)
    l_t = 10 ** l_t
    a_t = 10 ** a_t
    b_t = 10 ** b_t
    lin_transfer = cv2.merge([l_t, a_t, b_t])
    lin_transfer = np.array(lin_transfer, dtype=np.uint8)

    #  Converting to RGB space from Linear-LMS space
    reshaped_lin_transfer = lin_transfer.transpose(2, 0, 1).reshape(3, -1)
    rgb_transfer = lms_to_rgb @ reshaped_lin_transfer
    rgb_transfer = rgb_transfer.reshape(b_s, r_s, g_s).transpose(1, 2, 0).astype(np.uint8)

    return rgb_transfer
