#!/bin/bash

set -e

export URL=http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
export trainx=train-images-idx3-ubyte.gz
export trainy=train-labels-idx1-ubyte.gz
export testx=t10k-images-idx3-ubyte.gz
export testy=t10k-labels-idx1-ubyte.gz

for dset in ${trainx} ${trainy} ${testx} ${testy}
do
  wget ${URL}${dset} && gunzip ${dset}
done