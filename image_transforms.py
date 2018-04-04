import numpy as np
import cv2
import os


def get_file_names(directory):
    list_files = os.listdir(directory)
    list_files = [file for file in list_files if file[0] != '.']


def outline_menu(image, outlines=False):
    # Gaussian Blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Canny Edge Detection
    edged = cv2.Canny(image, 100, 200)

    # Dilation
    kernel = np.ones((8,8),np.uint8)
    dilation = cv2.dilate(edged,kernel,iterations = 1)

    # Hough Line Transform
    minLineLength = 100
    maxLineGap = 400
    #minLineLength = 50
    #maxLineGap = 80

    lines = cv2.HoughLinesP(dilation,1,np.pi/180,100,minLineLength,maxLineGap)
    if outlines:
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                cv2.line(image,(x1,y1),(x2,y2),(0,255,0),10)
        return image
    else:
        return lines


def convert_angle(degrees):
    if degrees < 0.0:
        return 180 + degrees
    else:
        return degrees


def get_deskew(image):
    lines = outline_menu(image, outlines=False)

    # if hough line transform finds no lines, returns None
    # if no lines, return image
    if lines is None:
        return image

    lines_reshaped = lines.reshape([lines.shape[0], lines.shape[-1]])

    angles = []
    for i, (x1, y1, x2, y2) in enumerate(lines_reshaped):
        angle = np.arctan2(y2 - y1, x2 - x1)
        angle = angle*180/np.pi
        angles.append(angle)
    #subsetting horizontal angles
    horizontal_angles = [angle for angle in angles if (angle > -27) and (angle < 27)]

    #subsetting vertical lines and angles
    vertical_lines = [line for line, angle in zip(lines_reshaped, angles) if (angle < -63) or (angle > 63)]
    vertical_angles = [angle for angle in angles if (angle < -63) or (angle > 63)]

    (h, w) = image.shape[:2]

    # If horizontal angles exist, use average horizontal angle as skew
    if len(horizontal_angles) > 0:
        skew = np.mean(horizontal_angles)

    # If there are no horizontal angles, use average of angles from distinct vertical lines to calculate skew
    # Assuming vertical lines separated more than 1/3rd of the width of the image are opposite menu borders
    elif (len(horizontal_angles) == 0) and (len(vertical_angles) > 0):
        skew = []
        (h, w) = image.shape[:2]
        for angle, line in zip(vertical_angles, vertical_lines):
            for compare_angle, compare_line in zip(vertical_angles, vertical_lines):
                if angle == compare_angle:
                    continue
                x_diff = abs((line[0] - compare_line[0]) + (line[2] - compare_line[2])) / 2
                angle_diff = abs(angle) - abs(compare_angle)

                if (x_diff > w/3) and (angle_diff < 10):
                    angle = convert_angle(angle)
                    compare_angle = convert_angle(compare_angle)
                    skew.append(np.mean([angle, compare_angle]))
        # If there are not multiple lines to compare, return the original image
        if len(skew) == 0:
            return image
        # If multiple lines, skew is 90 degree shift of average angle
        else:
            skew = np.mean(skew) - 90

    # If no horizontal or complementary vertical lines are found by Hough Line Transform, return original image
    else:
        return image
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
    flags=cv2.INTER_CUBIC)
    return rotated


def custom_crop(image):
    lines = outline_menu(image)

    if lines is None:
        return image

    lines_reshaped = lines.reshape([lines.shape[0], lines.shape[-1]])

    #calculates angles of lines
    angles = []
    for i, (x1, y1, x2, y2) in enumerate(lines_reshaped):
        angle = np.arctan2(y2 - y1, x2 - x1)
        angle = angle*180/np.pi
        angles.append(angle)

    # horizontal lines
    horizontal_lines = np.array([line for line, angle in zip(lines_reshaped, angles) if (angle > -27) and (angle < 27)])

    # vertical lines
    vertical_lines = np.array([line for line, angle in zip(lines_reshaped, angles) if (angle < -63) or (angle > 63)])

    (h, w) = image.shape[:2]

    # finding slicing bounds based on outermost lines
    if len(horizontal_lines) != 0:
        top_bound = np.min(horizontal_lines[:,3])
        bottom_bound = np.max(horizontal_lines[:,1])
    else:
        top_bound, bottom_bound = 0, h

    if len(vertical_lines) != 0:
        left_bound = np.min(vertical_lines[:,0])
        right_bound = np.max(vertical_lines[:,2])
    else:
        left_bound, right_bound = 0, w


    # compares bounds to each other to ensure they are not more or less on the same line
    # Then, compares them to edges to make sure the line is not random noise or text detected as an edge


    if (right_bound - left_bound) < w/2:
        if right_bound < w/1.33:
            right_bound = w
        if left_bound > w/3:
            left_bound = 0
    if (bottom_bound - top_bound < 0.75* h):
        if bottom_bound < 0.75 * h:
            bottom_bound = h
        if top_bound > h/4:
            top_bound = 0
    return image[top_bound:bottom_bound,left_bound:right_bound]
