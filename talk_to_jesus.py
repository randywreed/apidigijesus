import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
import configparser

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange

from pathlib import Path

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
config=configparser.ConfigParser()
config.read('model.ini')
model=config['DEFAULT']['model']
modeldir=config['DEFAULT']['dir']
port=config['DEFAULT']['port']
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelWithLMHead.from_pretrained(modeldir)
from flask import Flask
from flask_restful import Resource, Api, reqparse

app=Flask(__name__)
api=Api(app)


class question(Resource):
    
    def post(self):
        parser=reqparse.RequestParser()
        parser.add_argument('question',required=True)
        args=parser.parse_args()
        # Configs
        logger = logging.getLogger(__name__)

        MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
        MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

        # parser=argparse.ArgumentParser(description="talk to Digital Jesus")
        # parser.add_argument("text",type=str,help='message to send to digital Jesus')
        # args=parser.parse_args()
        # print(args.text)



        step=0
        new_user_input_ids = tokenizer.encode((f">> User:{args.question}") + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = new_user_input_ids
        chat_history_ids = model.generate(
                bot_input_ids, max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.92, top_k = 50
            )
        answer="Jesus: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
        return {'answer':answer}, 200

api.add_resource(question,'/question')

if __name__=='__main__':
    print('running on localhost port {}'.format(port))
    app.run(host="0.0.0.0", port=port)
