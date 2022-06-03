# cf. https://tech.unifa-e.com/entry/2018/09/20/183742

from facenet.src import facenet
import tensorflow as tf
import numpy as np
from PIL import Image

class FaceEmbedding(object):

    def __init__(self, model_path):
        # モデルを読み込んでグラフに展開
        facenet.load_model(model_path)
        
        self.input_image_size = 160
        #self.sess = tf.Session()
        self.sess = tf.compat.v1.Session()
        self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]
        
    def __del__(self):
        self.sess.close()
        
    def load_image(self, image_path, width, height, mode):
        image = Image.open(image_path)
        image = image.resize([width, height], Image.BILINEAR)
        return np.array(image.convert(mode))
        
    def face_embeddings(self, image_path):
        image = self.load_image(image_path, self.input_image_size, self.input_image_size, 'RGB')
        prewhitened = facenet.prewhiten(image)
        prewhitened = prewhitened.reshape(-1, prewhitened.shape[0], prewhitened.shape[1], prewhitened.shape[2])
        feed_dict = { self.images_placeholder: prewhitened, self.phase_train_placeholder: False }
        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return embeddings