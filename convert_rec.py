import mxnet as mx
import numpy as np
import sys, os

datasets = ['imdb', 'wiki', 'megaage'] # choose dataset: imdb, wiki, megaage 
train_vals = ['train', 'val'] # choose train or val to convert

for dataset in datasets:
  for train_val in train_vals:
    print('starting to convert %s %s' %(dataset,train_val))
    source_dir = '/media/3T_disk/my_datasets/ssr_net/%s'%dataset
    output_dir = '/media/3T_disk/my_datasets/imdb_wiki/%s'%dataset
    source_idx = os.path.join(source_dir, '%s.idx'%train_val)
    source_rec = os.path.join(source_dir, '%s.rec'%train_val)
    output_idx = os.path.join(output_dir,'%s.idx'%train_val)
    output_rec = os.path.join(output_dir,'%s.rec'%train_val)
    writer = mx.recordio.MXIndexedRecordIO(output_idx, output_rec, 'w')  
    imgrec = mx.recordio.MXIndexedRecordIO(source_idx, source_rec, 'r')  
    seq = list(imgrec.keys)
    widx = 0
    for img_idx in seq:
      if img_idx%1000==0:
        print('processing %s %d' %(train_val,img_idx))
      s = imgrec.read_idx(img_idx)
      header, img = mx.recordio.unpack(s)
      try:
        image = mx.image.imdecode(img).asnumpy()
      except:
        continue
      label = header.label
      age = int(label[0])
      gender = int(label[1])

      nheader = mx.recordio.IRHeader(0, [gender, age], widx, 0)
      rgb = image[:,:,::-1]
      s = mx.recordio.pack_img(nheader, rgb, quality=95, img_fmt='.jpg')
      writer.write_idx(widx, s)
      widx+=1


