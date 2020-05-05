import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(
    description='Parser file for Image Classifier',
)

parser.add_argument ('image_path', help = 'Path to the image you want to classify.', type = str)
parser.add_argument('saved_model', help = "Path to model", type = str)
parser.add_argument("--top_k", help="Return the top KK most likely classes", required = False, default = 5)
parser.add_argument("--category_names", help="Path to a JSON file mapping labels to flower names", required = False, default = "label_map.json")
         
def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()
      
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = np.expand_dims(process_image(test_image), axis=0)
    preds = model.predict(processed_test_image)[0]
    probs, class_idx = tf.math.top_k(preds, k=top_k)
    classes=[]
    for i in class_idx.numpy():
        classes.append(class_names[str(i)])
    print("Flower names: {} ".format(classes))
    print("Probabilities: {} ".format(probs.numpy()*100))
    return probs.numpy(), classes

args = parser.parse_args()
im_path = args.image_path
model = args.saved_model
k = int(args.top_k)
with open(args.category_names, 'r') as f:
    class_names = json.load(f)

loaded_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer': hub.KerasLayer})
predict(im_path, loaded_model, k)