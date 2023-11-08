import requests
import json

import logging
import ctypes
import os
from llama_cpp import Llama
from llama_cpp.llava_cpp import (clip_model_load, llava_image_embed_make_with_filename, llava_image_embed_make_with_bytes, llava_image_embed_free, llava_validate_embed_size, llava_eval_image_embed)

logger = logging.getLogger(__name__)

# delete after API implemented
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="./models/llava/ggml-model-q5_k.gguf")
parser.add_argument("-mp", "--mmproj", type=str, default="./models/llava/mmproj-model-f16.gguf")
parser.add_argument("-i", "--image", type=str, default=os.path.join(os.path.dirname(__file__), "images/", "overfitting_lc.png"))
parser.add_argument("-l", "--length_max_target", type=int, default=384)
parser.add_argument("-t", "--temp", type=float, default=0.1)
parser.add_argument("-sp", "--system_prompt", type=str, default="You are an image analysis expert. You give helpful and polite answers to questions. Your job is to describe the setting and context of an image, as concisely as possible.")
parser.add_argument("-up", "--user_prompt", type=str, default="Describe the setting and context of this image.")
args = parser.parse_args()

def llava_inference_local(
    model,
    mmproj,
    system_prompt,
    user_prompt,
    image,
    temp,
    length_max_target,
  ):

    print(f"loading llm model from {model}")
    if not os.path.exists(model):
        raise FileNotFoundError(model)
    llm = Llama(model_path=model, n_ctx=2048, n_gpu_layers=1) # longer context needed for image embeds

    print(f"loading clip model from {mmproj}")
    if not os.path.exists(mmproj):
        raise FileNotFoundError(mmproj)
    ctx_clip = clip_model_load(mmproj.encode('utf-8'), 1)

    if not llava_validate_embed_size(llm.ctx, ctx_clip):
        raise RuntimeError("llm and mmproj model embed size mismatch")

    print(f"loading image from {image}")
    if not os.path.exists(image):
        raise FileNotFoundError(image)
    image_embed = llava_image_embed_make_with_filename(ctx_clip=ctx_clip, n_threads=1, image_path=image.encode('utf8'))

    # eval system prompt
    llm.eval(llm.tokenize(f"SYSTEM PROMPT: {system_prompt}\n".encode('utf8')))
    llm.eval(llm.tokenize(f"USER PROMPT: {user_prompt}\n".encode('utf8')))

    # eval image embed
    n_past = ctypes.c_int(llm.n_tokens)
    n_past_p = ctypes.byref(n_past)
    llava_eval_image_embed(llm.ctx, image_embed, llm.n_batch, n_past_p)
    llm.n_tokens = n_past.value
    llava_image_embed_free(image_embed)

    llm.eval(llm.tokenize("\nASSISTANT: ".encode('utf8')))

    # get output
    completion = ''
    for i in range(length_max_target):
        t_id = llm.sample(temp=temp)
        try:
            t = llm.detokenize([t_id]).decode('utf8')
        except UnicodeDecodeError:
            t = llm.detokenize([t_id]).decode('iso-8859-1')
        if t == "</s>" or t == "ë":
            break
        completion += t
        llm.eval([t_id])
    
    print("\nDONE with LOCAL llava image inference\n")
    print(completion)
    return completion

# llava_inference_local(args.model, args.mmproj, args.system_prompt, args.user_prompt, args.image, args.temp, args.length_max_target)

# TODO: update this to a true API after llama.cpp uptakes llava changes as an API
def llava_inference_webserver(
    url="http://localhost:5003/v1/chat/completions",
    system_prompt="You are an image analysis expert. You give helpful and polite answers to questions. Your job is to describe the setting and context of an image, as concisely as possible.\n",
    user_prompt="Describe the setting and context of this image.",
    image_path=os.path.join(os.path.dirname(__file__), "images/", "overfitting_lc.png"),
    temperature=0.1,
    max_tokens=384,
  ):
  ctx_clip = clip_model_load("./models/llava/mmproj-model-f16.gguf".encode('utf8'), 1)
  image_embed = llava_image_embed_make_with_filename(ctx_clip=ctx_clip, n_threads=1, image_path=image_path.encode('utf8'))

  headers = {
    'Content-Type': 'application/json'
  }
  payload = json.dumps({
    "messages": [
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": f"{user_prompt} {image_embed}"
      }
    ],
    "max_tokens": max_tokens,
    "temperature": temperature,
    "stream": False
  })

  response = requests.request("POST", url, headers=headers, data=payload)
  response = response.json()
  completion = response['choices'][0]['message']['content'].strip()

  print("\nDONE with WEBSERVER llava image inference\n")
  print(completion)
  return completion

# llava_inference_webserver(args.system_prompt, args.user_prompt, args.image, args.temp, args.length_max_target)

# # llm direct (local) call

# from llama_cpp import Llama

# llm_q4 = Llama(model_path="/Users/sagu/Documents/SideProjects/reels-to-text/models/7B/llama-2-7b.Q4_K_M.gguf", n_gpu_layers=1)
# llm_q5_ks = Llama(model_path="/Users/sagu/Documents/SideProjects/reels-to-text/models/7B/llama-2-7b.Q5_K_S.gguf", n_gpu_layers=1)
# llm_q5_km = Llama(model_path="/Users/sagu/Documents/SideProjects/reels-to-text/models/7B/llama-2-7b.Q5_K_M.gguf", n_gpu_layers=1)

# output = llm_q5_km("The names of the planets in our solar system are: ", max_tokens=256, stop=["Q:", "\n"], echo=True)
# print(output)


# # llm server (local) call

# response = completion(
#     messages=[{"content": "The names of the planets in our solar system are: ","role": "user"}],
# )
# print(response)


# llm server (local) call 2

# ./main -ngl 32 -m llama-2-7b.q4_K_M.gguf --color -c 4096 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "{prompt}"

# Change -ngl 32 to the number of layers to offload to GPU. Remove it if you don't have GPU acceleration.

# Change -c 4096 to the desired sequence length. For extended sequence models - eg 8K, 16K, 32K - the necessary RoPE scaling parameters are read from the GGUF file and set by llama.cpp automatically.

# If you want to have a chat-style conversation, replace the -p <PROMPT> argument with -i -ins

# For other parameters and how to use them, please refer to the llama.cpp documentation

