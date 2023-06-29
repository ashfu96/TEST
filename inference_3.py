import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import glob
import matplotlib.pyplot as plt
import random
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

################################## INFERENCE ###########################

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes dovrebbe essere di tipo int.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Gestisci i modelli con maschere:
    if 'detection_masks' in output_dict:
        # Reframe the the box mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()   

    return output_dict


def run_inference(model, category_index, image_path):
    if os.path.isdir(image_path):
        image_paths = []
        for file_extension in ('*.png', '*jpg'):
            image_paths.extend(glob.glob(os.path.join(image_path, file_extension)))

        """add iterator here"""
        i = 0
        for i_path in image_paths:
            image_np = load_image_into_numpy_array(i_path)
            # Rilevazione effettiva.
            output_dict = run_inference_for_single_image(model, image_np)

########### THRESHOLD #############
            threshold = 0.3 

            found_objects = {}
            object_count = 0

            #vis_util.draw_bounding_box_on_image_array(image_np,y_min,x_min,y_max,x_max,color='red',thickness=6,use_normalized_coordinates=True)
            for x, (y_min, x_min, y_max, x_max) in enumerate(output_dict['detection_boxes']):
                if output_dict['detection_scores'][x] > threshold:            
                    current_label = category_index[output_dict['detection_classes'][x]]['name']
                    if current_label in found_objects:
                        found_objects[current_label] += 1  # se esiste già, incrementa
                    else:
                        found_objects[current_label] = 1  #  altrimenti, aggiungi e imposta a 1
                    object_count += 1
            print(f'Totale oggetti trovati in {i_path}: {object_count}')
            print(found_objects)

            # vis_util.draw_bounding_box_on_image_array(image_np,
            #                             y_min,
            #                             x_min,
            #                             y_max,
            #                             x_max,
            #                             color='red',
            #                             thickness=6,
            #                             #display_str_list=(),
            #                             use_normalized_coordinates=True)
# vis_util.draw_bounding_box_on_image_array(image_np,y_min,x_min,y_max,x_max,color='red',thickness=6,use_normalized_coordinates=True)

            # Visualizzazione dei risultati della rilevazione.
            draw_boxes_on_image(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                #line_thickness=6,
                min_score_thresh=threshold
                )
            
            plt.imshow(image_np)
            plt.savefig("outputs/detection_output{}.png".format(i))  # make sure to make an outputs folder
            i = i + 1


######################################################

def draw_boxes_on_image(image_np, detection_boxes, detection_classes, detection_scores, category_index,
                        instance_masks=None, use_normalized_coordinates=True, min_score_thresh=0.3):
    colors = []
    # Convert detection boxes to absolute coordinates if normalized
    if use_normalized_coordinates:
        height, width, _ = image_np.shape
        detection_boxes = np.multiply(detection_boxes, [height, width, height, width])

    for box, cls, score in zip(detection_boxes, detection_classes, detection_scores):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    # Loop through each detected object
    for box, cls, score, color in zip(detection_boxes, detection_classes, detection_scores, colors):
        # Check if the detection score is above the minimum threshold
        if score >= min_score_thresh:
            ymin, xmin, ymax, xmax = box.astype(int)

            # Draw the bounding box on the image
            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, 2)

            # Get the class label
            class_label = category_index[cls]['name']

            # Format the label with the class name and score
            label = f'{class_label}: {round(score * 100, 2)}%'

            # Adjust the font size dynamically to fit within the bounding box
            font_scale = 0.56
            thickness = 1

            # Calculate the initial width and height of the label
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            while label_width > (xmax - xmin) or label_height > (ymax - ymin):
                font_scale -= 0.1

                # Break the loop if font size becomes too small
                if font_scale <= 0.4:
                    break

                # Recalculate the label width and height
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Calculate the position of the label
            label_origin = (xmin, ymin - 10) if ymin >= label_height + 10 else (xmin, ymin + label_height + 10)

            # Draw the label background
            cv2.rectangle(image_np, (xmin, label_origin[1] - label_height),
                          (xmin + label_width, label_origin[1]), color, cv2.FILLED)

            # Draw the label text
            cv2.putText(image_np, label, (label_origin[0] + 5, label_origin[1] - 5),
                        cv2.FONT_ITALIC, font_scale, (0, 0, 0), thickness)

    for box, cls, score, color in zip(detection_boxes, detection_classes, detection_scores, colors):
        # Check if the detection score is above the minimum threshold
        if score >= min_score_thresh:
            ymin, xmin, ymax, xmax = box.astype(int)

            # Get the class label
            class_label = category_index[cls]['name']

            # Format the label with the class name and score
            label = f'{class_label}: {round(score * 100, 2)}%'

            # Adjust the font size dynamically to fit within the bounding box
            font_scale = 0.56
            thickness = 1

            # Calculate the initial width and height of the label
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            while label_width > (xmax - xmin) or label_height > (ymax - ymin):
                font_scale -= 0.1

                # Break the loop if font size becomes too small
                if font_scale <= 0.4:
                    break

                # Recalculate the label width and height
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Calculate the position of the label
            label_origin = (xmin, ymin - 10) if ymin >= label_height + 10 else (xmin, ymin + label_height + 10)

            # Draw the label background
            cv2.rectangle(image_np, (xmin, label_origin[1] - label_height),
                          (xmin + label_width, label_origin[1]), color, cv2.FILLED)

            # Draw the label text
            cv2.putText(image_np, label, (label_origin[0] + 5, label_origin[1] - 5),
                        cv2.FONT_ITALIC, font_scale, (0, 0, 0), thickness)


    return image_np

#######################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to image (or folder)')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)

    run_inference(detection_model, category_index, args.image_path)
