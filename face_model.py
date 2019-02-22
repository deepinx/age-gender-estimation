from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from essh_detector import ESSHDetector
from mtcnn_detector import MtcnnDetector
import face_image
import face_preprocess


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    if args.gpu>=0:
      ctx = mx.gpu(args.gpu)
    else:
      ctx = mx.cpu()
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')

    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = ESSHDetector(prefix='./ssh-model/essh', epoch=0, ctx_id=args.gpu, test_mode=False)
      # detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector


  def get_input(self, face_img, args):
    if args.det==0:
      ret = self.detector.detect_face(face_img, det_type = self.args.det)
      if ret is None:
        return None
      bbox, points = ret
      if bbox.shape[0]==0:
        return None
      bbox = bbox[:,0:4]
      points = points[:,:].reshape((-1,2,5))
      points = np.transpose(points, (0,2,1))
    else:
      ret = self.detector.detect(face_img, 0.5, scales=[1.0])
      if ret is None:
        return None
      bbox = ret[:,0:4]
      points = ret[:,5:15].reshape((-1,5,2))

    #print(bbox)
    #print(points)
    input_blob = np.zeros((bbox.shape[0], 3, 112, 112))
    for i in range(bbox.shape[0]):
      nimg = face_preprocess.preprocess(face_img, bbox[i,:], points[i,:], image_size='112,112')
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      # input_blob = np.expand_dims(aligned, axis=0)
      input_blob[i,:] = aligned
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))

    return db, bbox, points


  def get_ga(self, data):
    self.model.forward(data, is_train=False)
    ret = self.model.get_outputs()[0].asnumpy()
    g = ret[:,0:2]
    gender = np.argmax(g, axis=1)
    a = ret[:,2:202].reshape((-1,100,2))
    a = np.argmax(a, axis=2)
    age = np.sum(a, axis=1)

    return gender, age

