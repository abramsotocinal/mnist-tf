import struct as st
import numpy as np

data_dir = 'data/'
class Idx2Np:
  def __init__(self,target='train',data_dir=data_dir):
    self.data_dir = data_dir
    self.fname = { 'train': self.data_dir+'train-images-idx3-ubyte',
          'train_label': self.data_dir+'train-labels-idx1-ubyte',
          'test': self.data_dir+'t10k-images-idx3-ubyte',
          'test_label': self.data_dir+'t10k-labels-idx1-ubyte'}
    if target not in self.fname.keys():
      raise ValueError('Please specify one of the following targets: {}'.format(self.fname.keys()))
    self.img, self.lbl = open(self.fname[target], 'rb'), open(self.fname[target+'_label'], 'rb')
    self.target = target
  def unpack(self):
    for i in (self.target,self.target+'_label'):
      self.f = self.img if i == self.target else self.lbl
      self.f.seek(0)
      self.magic = st.unpack('>BBBB',self.f.read(4))
      self.total = st.unpack('>I',self.f.read(4))[0] #num of images
      if i == self.target:
        self.nR = st.unpack('>I',self.f.read(4))[0] #num of rows
        self.nC = st.unpack('>I',self.f.read(4))[0] #num of columns
      self._to_array() if i == self.target else self._to_array_label()
  def _to_array(self):
    self.array = np.zeros((self.total,self.nC,self.nR))
    self.bytes_total = self.total*self.nC*self.nR*1
    #self.array = 255 - np.asarray(st.unpack('>I'*self.bytes_total, self.f.read(self.bytes_total))).reshape((self.total,self.nR,self.nC))
    self.array = np.fromfile(self.f, dtype=np.dtype(np.uint8)).newbyteorder(">").reshape(self.total,self.nC,self.nR)
    self.f.close()
  def _to_array_label(self):
    self.label_array = np.fromfile(self.f, dtype=np.dtype(np.uint8)).newbyteorder('>')
    self.f.close()
