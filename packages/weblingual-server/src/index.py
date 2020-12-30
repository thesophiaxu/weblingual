import collections
import csv
import json
import os
import pickle
import sys
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from transformers.pipelines import Pipeline, TokenClassificationArgumentHandler, ArgumentHandler, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
from transformers.configuration_utils import PretrainedConfig
from transformers.data import SquadExample, SquadFeatures, squad_convert_examples_to_features
from transformers.file_utils import add_end_docstrings, is_tf_available, is_torch_available, requires_pandas
from transformers.modelcard import ModelCard
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.utils import logging

import torch

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTableQuestionAnswering,
    AutoModelForTokenClassification,
)

import numpy as np


from flask import Flask, escape, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
from transformers import pipeline, GPT2Model, AutoTokenizer, GPT2Tokenizer, GPT2ForSequenceClassification, BertModel, AutoModelForTokenClassification, BertTokenizer, BertForTokenClassification
import json
# Environmenal Variables
VERSION = "0.0.1"










#Pipeline thingy
class TokenClassificationPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using any :obj:`ModelForTokenClassification`. See the `named entity recognition
    examples <../task_summary.html#named-entity-recognition>`__ for more information.

    This token recognition pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location
    or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=token-classification>`__.
    """

    default_input_names = "sequences"

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = TokenClassificationArgumentHandler(),
        device: int = -1,
        binary_output: bool = False,
        ignore_labels=["O"],
        task: str = "",
        grouped_entities: bool = False,
        ignore_subwords: bool = False,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            device=device,
            binary_output=binary_output,
            task=task,
        )

        self.check_model_type(
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
        )

        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self._args_parser = args_parser
        self.ignore_labels = ignore_labels
        self.grouped_entities = grouped_entities
        self.ignore_subwords = ignore_subwords

        if self.ignore_subwords and not self.tokenizer.is_fast:
            raise ValueError(
                "Slow tokenizers cannot ignore subwords. Please set the `ignore_subwords` option"
                "to `False` or use a fast tokenizer."
            )

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with
            :obj:`grouped_entities=True`) with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
              `grouped_entities` is set to True.
            - **index** (:obj:`int`, only present when ``self.grouped_entities=False``) -- The index of the
              corresponding token in the sentence.
            - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
            - **end** (:obj:`int`, `optional`) -- The index of the end of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
        """

        inputs, offset_mappings = self._args_parser(inputs, **kwargs)

        answers = []

        for i, sentence in enumerate(inputs):
            # Manage correct placement of the tensors
            with self.device_placement():

                tokens = self.tokenizer(
                    sentence,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    truncation=True,
                    padding=True,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=self.tokenizer.is_fast,
                )
                if self.tokenizer.is_fast:
                    offset_mapping = tokens.pop("offset_mapping").cpu().numpy()[0]
                elif offset_mappings:
                    offset_mapping = offset_mappings[i]
                else:
                    offset_mapping = None

                special_tokens_mask = tokens.pop("special_tokens_mask").cpu().numpy()[0]

                # Forward
                if self.framework == "tf":
                    entities = self.model(tokens.data)[0][0].numpy()
                    input_ids = tokens["input_ids"].numpy()[0]
                else:
                    with torch.no_grad():
                        tokens = self.ensure_tensor_on_device(**tokens)
                        entities_all = self.model(**tokens)[0].cpu().numpy()
                        input_ids_all = tokens["input_ids"].cpu().numpy()

            for i in range(len(sentence)):
                entities = entities_all[i]
                input_ids = input_ids_all[i]


                score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
                labels_idx = score.argmax(axis=-1)

                entities = []
                # Filter to labels not in `self.ignore_labels`
                # Filter special_tokens
                filtered_labels_idx = [
                    (idx, label_idx)
                    for idx, label_idx in enumerate(labels_idx)
                    if (self.model.config.id2label[label_idx] not in self.ignore_labels) and not special_tokens_mask[idx]
                ]

                for idx, label_idx in filtered_labels_idx:
                    if offset_mapping is not None:
                        start_ind, end_ind = offset_mapping[idx]
                        word_ref = sentence[start_ind:end_ind]
                        word = self.tokenizer.convert_ids_to_tokens([int(input_ids[idx])])[0]
                        is_subword = len(word_ref) != len(word)

                        if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                            word = word_ref
                            is_subword = False
                    else:
                        word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))

                        start_ind = None
                        end_ind = None

                    entity = {
                        "word": word,
                        "score": score[idx][label_idx].item(),
                        "entity": self.model.config.id2label[label_idx],
                        "index": idx,
                        "start": start_ind,
                        "end": end_ind,
                    }

                    if self.grouped_entities and self.ignore_subwords:
                        entity["is_subword"] = is_subword

                    entities += [entity]

                if self.grouped_entities:
                    answers += [self.group_entities(entities)]
                # Append ungrouped entities
                else:
                    answers += [entities]


        if len(answers) == 1:
            return answers[0]
        return answers

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        if entities:
            last_idx = entities[-1]["index"]

        for entity in entities:

            is_last_idx = entity["index"] == last_idx
            is_subword = self.ignore_subwords and entity["is_subword"]
            if not entity_group_disagg:
                entity_group_disagg += [entity]
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
                continue

            # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" suffixes
            # Shouldn't merge if both entities are B-type
            if (
                (
                    entity["entity"].split("-")[-1] == entity_group_disagg[-1]["entity"].split("-")[-1]
                    and entity["entity"].split("-")[0] != "B"
                )
                and entity["index"] == entity_group_disagg[-1]["index"] + 1
            ) or is_subword:
                # Modify subword type to be previous_type
                if is_subword:
                    entity["entity"] = entity_group_disagg[-1]["entity"].split("-")[-1]
                    entity["score"] = np.nan  # set ignored scores to nan and use np.nanmean

                entity_group_disagg += [entity]
                # Group the entities at the last entity
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
            # If the current entity is different from the previous entity, aggregate the disaggregated entity group
            else:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
                entity_group_disagg = [entity]
                # If it's the last entity, add it to the entity groups
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]

        return entity_groups

















# Models and Pipelines
#model = GPT2ForTokenClassification.from_pretrained('gpt2-large')
#model_classification = GPT2ForSequenceClassification.from_pretrained('gpt2-large')
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
#model_token_classification = BertForTokenClassification.from_pretrained('bert-base-uncased')
#model_classification = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#pipeline_sentiment = pipeline('sentiment-analysis', model=model_classification, tokenizer=tokenizer)
#pipeline_ner = pipeline('ner', model=model_token_classification, tokenizer=tokenizer)
model_ner = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english").to('cuda')
tokenizer_ner = AutoTokenizer.from_pretrained('bert-large-cased', padding=True)
pipeline_ner = TokenClassificationPipeline(task="ner", model=model_ner, tokenizer=tokenizer_ner, grouped_entities=True, device=0)
#pipeline_ner = pipeline('ner', grouped_entities=True)


# Handlers
def handle_ner(texts):
    return pipeline_ner(texts)

def handle_sentiment(text):
    #return pipeline_sentiment(text, padding=True)
    pass

@app.route('/')
def root():
    return '''WebLingual server version {}
APIs:
- /tasks/ner POST JSON
    Accepts a JSON object like \{"text": "Paris is the capital of France"\}, where the text field is the text to determine.
    Returns a JSON object like \{"result": ...\}, where result field is the result of the task.
'''.format(VERSION)

@app.route('/tasks/sentiment', methods=['POST'])
def sentiment_analysis():
    text = request.get_json(force = True)['text']
    print(text)
    result = handle_sentiment(text=text)
    return json.dumps(str({"result": result[0]}))

@app.route('/tasks/ner', methods=['POST'])
def ner_task():
    texts = request.get_json(force = True)['texts']
    print(texts)
    result = handle_ner(texts=texts)
    return json.dumps(str({"result": result}))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=22222)