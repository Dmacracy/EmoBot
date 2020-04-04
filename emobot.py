import os
import re
import sys
import json
import time
import random
import datetime
import requests

import cv2
import twitter
import numpy as np
import wikipedia as wik
import gpt_2_simple as gpt2

def train_gpt2_model(fileName):
  model_name = "124M"
  if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
    

  sess = gpt2.start_tf_sess()
  gpt2.finetune(sess,
              dataset=fileName,
              model_name=model_name,
              steps=1000,
              restore_from='fresh',
              run_name='run'+fileName,
              print_every=50,
              sample_every=200,
              save_every=500)


def gen_text(modelName, length=50, prefix=None):
  sess = gpt2.start_tf_sess()
  gpt2.load_gpt2(sess, run_name=modelName)
  if prefix:
    gpt2.generate(sess,
                  length=length,
                  temperature=0.7,
                  prefix=prefix,
                  nsamples=5,
                  batch_size=5)
    
  else:
    gpt2.generate(sess, length=length)

if __name__ == "__main__":
  fileName = sys.argv[1]
  modelName = 'run'+fileName
  if len(sys.argv) > 2:
    prefix = " ".join(sys.argv[2:])
  #train_gpt2_model(fileName)
  if prefix:
    gen_text(modelName, prefix=prefix)
  else:
    gen_text(modelName)

    
    


