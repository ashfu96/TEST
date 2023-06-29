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
    class_colors = {
        1: (255, 0, 0),   # Crepa: Rosso
        2: (0, 255, 0),   # Corrosione: Verde
        3: (0, 0, 255),   # Chiazze: Blu
        4: (255, 255, 0), # Superficie_bucherellata: Giallo
        5: (255, 0, 255), # Scaglie_laminate: Magenta
        6: (0, 255, 255)  # Graffi: Ciano
    }

    if use_normalized_coordinates:
        height, width, _ = image_np.shape
        detection_boxes = np.multiply(detection_boxes, [height, width, height, width])

    for box, cls, score in zip(detection_boxes, detection_classes, detection_scores):
        if score >= min_score_thresh:
            ymin, xmin, ymax, xmax = box.astype(int)
            class_label = category_index[cls]['name']
            color = class_colors[cls]
            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{class_label}: {round(score * 100, 2)}%'
            font_scale = 0.56
            thickness = 1
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            while label_width > (xmax - xmin) or label_height > (ymax - ymin):
                font_scale -= 0.1
                if font_scale <= 0.4:
                    break
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            label_origin = (xmin, ymin - 10) if ymin >= label_height + 10 else (xmin, ymin + label_height + 10)
            cv2.rectangle(image_np, (xmin, label_origin[1] - label_height),
                          (xmin + label_width, label_origin[1]), color, cv2.FILLED)
            cv2.putText(image_np, label, (label_origin[0] + 5, label_origin[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    legend_height = 30
    legend_width = image_np.shape[1]
    legend_img = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

    for idx, cls in enumerate(category_index):
        class_label = category_index[cls]['name']
        color = class_colors[cls]
        cv2.rectangle(legend_img, (idx * (legend_width // len(category_index)), 0),
                      ((idx + 1) * (legend_width // len(category_index)), legend_height), color, cv2.FILLED)
        cv2.putText(legend_img, class_label, (idx * (legend_width // len(category_index)) + 5, legend_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    
    image_np = np.concatenate((image_np, legend_img), axis=0)

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
