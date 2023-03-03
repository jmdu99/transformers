import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from neural_compressor.config import PostTrainingQuantConfig
from optimum.intel.neural_compressor import INCQuantizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help='Name or path of the pretrained model')
parser.add_argument('--save_dir', type=str, required=True, help='Directory where the quantized model will be saved')
parser.add_argument('--approach', type=str, default='dynamic', help='Approach for post-training quantization')

args = parser.parse_args()

model_name = args.model_name
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
save_dir = args.save_dir
AutoTokenizer.from_pretrained(model_name).save_pretrained(save_dir)

quantization_config = PostTrainingQuantConfig(approach=args.approach)
quantizer = INCQuantizer.from_pretrained(model)

quantizer.quantize(quantization_config=quantization_config, save_directory=save_dir)

"""
python quantize.py 
    --model_name 'outputs/bart-large/6-3/fine-tuned' 
    --save_dir 'outputs/bart-large/6-3/fine-tuned-quantized' 
    --approach dynamic
"""
