import numpy as np
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import glob
import matplotlib.pyplot as plt

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

#lista vuota per salvare le predizioni
prediction_results = []

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model


def load_image_into_numpy_array(path):
    """Carica un'immagine da un file in un array numpy.
    Mette l'immagine in un array numpy da alimentare nel grafo tensorflow.
    Nota che per convenzione la mettiamo in un array numpy con forma
    (altezza, larghezza, canali), dove canali=3 per RGB.
    Args:
      path: un percorso del file (può essere locale o su Colossus)
    Returns:
      array numpy uint8 con forma (altezza_img, larghezza_img, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    # L'input deve essere un tensore, convertilo usando `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # Il modello si aspetta un batch di immagini, quindi aggiungi un asse con `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # Tutte le uscite sono tensori batch.
    # Convertile in array numpy e prendi l'indice [0] per rimuovere la dimensione del batch.
    # Siamo interessati solo ai primi num_detections
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes dovrebbe essere di tipo int.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Gestisci i modelli con maschere:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
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

            threshold = 0.5
            found_objects = {}
            object_count = 0
            for x, (y_min, x_min, y_max, x_max) in enumerate(output_dict['detection_boxes']):
                if output_dict['detection_scores'][x] > threshold:
                    current_label = category_index[output_dict['detection_classes'][x]]['name']
                    if current_label in found_objects:
                        found_objects[current_label] += 1  # se esiste già, incrementa
                    else:
                        found_objects[current_label] = 1  #  altrimenti, aggiungi e imposta a 1
                    object_count += 1
            print(f'Total objects found in {i_path}: {object_count}')
            print(found_objects)
            
            prediction_results.append({
                'image_path': i_path,
                'object_count': object_count,
                'found_objects': found_objects
            })
            

            # Visualizzazione dei risultati della rilevazione.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            """The existing plt lines do not work on local pc as they are not setup for GUI
                Use plt.savefig() to save the results instead and view them in a folder"""
            plt.imshow(image_np)
            # plt.show()
            plt.savefig("outputs/detection_output{}.png".format(i))  # make sure to make an outputs folder
            i = i + 1
            

    return prediction_results

    # else:
    #     image_np = load_image_into_numpy_array(image_path)
    #     # Actual detection.
    #     output_dict = run_inference_for_single_image(model, image_np)
    #     # Visualization of the results of a detection.
    #     vis_util.visualize_boxes_and_labels_on_image_array(
    #         image_np,
    #         output_dict['detection_boxes'],
    #         output_dict['detection_classes'],
    #         output_dict['detection_scores'],
    #         category_index,
    #         instance_masks=output_dict.get('detection_masks_reframed', None),
    #         use_normalized_coordinates=True,
    #         line_thickness=8)
    #     plt.imshow(image_np)
    #     plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    parser.add_argument('-i', '--image_path', type=str, required=True, help='Path to image (or folder)')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)
    prediction_results = run_inference(detection_model, category_index, args.image_path) #save_outputs=true

    run_inference(detection_model, category_index, args.image_path)
