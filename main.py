import logging
import getopt, sys
import os, glob
import random
from pydantic import BaseModel
from typing import Optional
import json
from datetime import datetime, timezone
from reels.download import download_clip, move_files
from reels.extract_audio import extract_audio
from reels.extract_setting import llava_inference_local, llava_inference_webserver
from reels.extract_video import extract_video
from reels.extract_metadata import extract_metadata
from reels.text_operations import Platform, clean_video_link, text_cleanup, call_gpt_api, call_gpt_api_simple, coherence_score, text_combine
from reels.extract_setting import llava_inference_local, llava_inference_webserver
import openai
from pinecone_ops import embedding_dimension, max_token_length_embedding, get_token_count, get_embedding_inner, get_embedding, pinecone_index_health, pinecone_upsert, pinecone_query, pinecone_query_filter
from mongodb_ops import create_client, insert_document
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# logging.basicConfig(level=logging.INFO)
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger("main")

# # TODO: figure out this logger filtering stuff (temp solution is change to INFO in `logging.conf``)
# class LoggerFilter(logging.Filter):
#     def filter(self, record):
#         return record.name in ['', 'main', 'reels']
# logger_filter = LoggerFilter()
# logger.addFilter(logger_filter)
# logger.filter(logger_filter)

# test queue:
# good results: 'theloverspassport_Cvus8p-PSgy', 'sanjosefoos_Coc8ySXDuPE', 'bradjbecca_CoFkxLps6Mx'
# mixed results: 'christiancruzfitness_CqrKsf0vdK5', 'scheereddzz_Cp_V4qwr7rg', 'suzionthemove_Cs9Ji3BOkl_'
# bad results: 'jaredemanuele_Co6EqedPMrr'

# global constants
PROJECT_VERSION = 0.0

# keys
openai.api_key = os.getenv("OPENAI_API_KEY")

# clients
openai_client = openai.OpenAI()
mdb_client = create_client()

# openai constants
OPENAI_MODEL = "gpt-3.5-turbo"
MODEL_TOKEN_LIMIT = 4096
MODEL_TOKEN_LIMIT_16K = 16385

# video_id = 'Co6EqedPMrr'
frame_dropout = 5

# Substitue below for start/end seconds
audio_path = f'./audio'
start = 0
end = 90

sample_images_path = f'./sample_images'
frames_path = f'./frames'
source_path = './'
videos_path = './videos'

# TODO (v1): add auth and user account db
# TODO (v1): figure out UI/UX section and integration (messenger bot?)

app = FastAPI(debug=True)
app.mount("/reels/frontend", StaticFiles(directory="reels/frontend"))

class SearchTextInput(BaseModel):
    searchText: str

class ReelIdInput(BaseModel):
    reelId: str

# TODO: convert code to async/await
@app.get('/', response_class=HTMLResponse)
def index():
    return open("reels/frontend/index.html").read()

@app.post('/inference/')
def inference_api(data: SearchTextInput):
    search_text = data.searchText
    logging.info({"message": f"FastAPI: Running inference with search text: {search_text}"})
    return (inference(search_text))

@app.post('/train/')
def train_api(data: ReelIdInput):
    reel_ids = data.reelId
    logging.info({"message": f"FastAPI: Running training with Video IDs: {reel_ids}"})
    return (train_outer(reel_ids))

def inference(query: str):
    assert get_token_count(query, OPENAI_MODEL) < max_token_length_embedding
    query_embedding = get_embedding_inner(openai_client, query)
    vectors = pinecone_query(query_embedding=query_embedding, namespace="reel", top_k=6)
    logging.info(f"retrieved vector dump: {vectors}")
    # TODO: come up with some algorithm to combine mode / cumulative rank / top_p?
    video_ids = [match['metadata']['id'] for match in vectors['matches']]
    # video_ids_set = set(video_ids)
    video_ids_cleaned = []
    for id in video_ids:
        if id[:9] == "YTSHORTS_" and id[-4:] == ".mp4":
            video_ids_cleaned.append(id[9:-4])
    logging.info(f"retrieved video ids: {video_ids_cleaned}")
    video_links = [f"https://www.youtube.com/embed/{id}" for id in video_ids_cleaned]

    return HTMLResponse(content=json.dumps({"videoLinks": video_links}), status_code=200)

def train_outer(video_links: str):
    video_links = clean_video_link(video_links.split(','))
    # TODO (v1): replace with asyncio or some other batch/scheduling utility
    train_responses = []
    for (platform, video_id) in video_links:
        train_response = train(platform=platform, video_id=video_id)
        train_responses.append(train_response)
        logging.info(f"training result for video_id {video_id}: HTTP {train_response.status_code}")
    return HTMLResponse(status_code=200)

def train(platform: Platform, video_id: str):
    # TODO: change to `namespace=platform.const` once those pinecone updates are done
    # embedding already exists in vector db
    vectors = pinecone_query_filter(query_embedding=[0.0]*embedding_dimension, namespace="reel", id=f'{platform.const}_{video_id}.mp4', top_k=1)
    if vectors['matches']:
        logging.info(f"training aborted - video {video_id} has already been upserted in vector db")
        return HTMLResponse(status_code=200)

    # TODO (v1): if file already exists, this method fucks up, debug it
    # download media if not already present and move it to correct folder
    video_id = download_clip(platform, video_id, source_path, videos_path) if not os.path.isfile(os.path.join(videos_path, f'{platform.const}_{video_id}.mp4')) else f'{platform.const}_{video_id}.mp4'

    # extract audio and transcribe
    # TODO (v1): save audio files in a better place
    logger.info(f'EXTRACTING AUDIO from {videos_path}/{video_id}')
    text_audio = extract_audio(video_id, videos_path, start, end, audio_path,)
    logger.info(f'extracted + cleaned audio: {text_audio}')
    # TODO: figure out `UnboundLocalError: local variable 'OPENAI_MODEL' referenced before assignment`
    # extra 170 tokens to account for system and user prompts
    token_count = get_token_count(str(text_audio)) + 170
    total_token_limit = int(1.8*token_count)
    # if longer context window is needed
    OPENAI_MODEL = "gpt-3.5-turbo-16k" if total_token_limit > MODEL_TOKEN_LIMIT else  "gpt-3.5-turbo"
    max_tokens = int(0.75*token_count)
    # max_tokens = int(0.5*MODEL_TOKEN_LIMIT_16K) if total_token_limit > MODEL_TOKEN_LIMIT else int(0.7*MODEL_TOKEN_LIMIT)
    logger.info(f'token in {token_count}; token total model limit {MODEL_TOKEN_LIMIT}; token total limit {total_token_limit}; token limit out {max_tokens}; model {OPENAI_MODEL}')
    text_audio = call_gpt_api(client=openai_client, text=text_audio, user_prompt_supplemental="", format="AUDIO", model=OPENAI_MODEL, max_tokens=max_tokens) if text_audio else ''
    logger.info(f'gpt api-d + cleaned audio: {text_audio}')
    text_audio = text_audio if text_audio and coherence_score(text_audio) > 0.725 else ''

    # extract text from video
    logger.info(f'EXTRACTING VIDEO from {videos_path}/{video_id}')
    text_video = extract_video(video_id, videos_path, frames_path, frame_dropout,)
    logger.info(f'extracted + cleaned video: {text_video}')
    text_video = text_cleanup(text_video) if text_video else ''
    logger.info(f'cleaned + statistic-ed video: {text_video}')
    # TODO: figure out `UnboundLocalError: local variable 'OPENAI_MODEL' referenced before assignment`
    # extra 200 tokens to account for system and user prompts
    token_count = get_token_count(str(text_video)) + 200
    total_token_limit = int(1.8*token_count)
    # if longer context window is needed
    OPENAI_MODEL = "gpt-3.5-turbo-16k" if total_token_limit > MODEL_TOKEN_LIMIT else  "gpt-3.5-turbo"
    max_tokens = int(0.75*token_count)
    # max_tokens = int(0.5*MODEL_TOKEN_LIMIT_16K) if total_token_limit > MODEL_TOKEN_LIMIT else int(0.7*MODEL_TOKEN_LIMIT)
    logger.info(f'token in {token_count}; token total model limit {MODEL_TOKEN_LIMIT}; token total limit {total_token_limit}; token limit out {max_tokens}; model {OPENAI_MODEL}')    
    text_video = call_gpt_api(client=openai_client, text=str(text_video), user_prompt_supplemental="", format="VIDEO", model=OPENAI_MODEL, temperature=0.8, max_tokens=max_tokens) if text_video else ''
    logger.info(f'gpt api-d + cleaned video: {text_video}')
    text_video = text_video if text_video and coherence_score(text_video) > 0.725 else ''
    
    # extract setting from three random frames of video
    video_settings = []
    frame_paths = glob.glob(os.path.join(frames_path, video_id) + "/*.png")
    frame_paths_selected = random.sample(frame_paths, 3)
    for frame in frame_paths_selected:
        setting = llava_inference_local(
            model="models/llava/ggml-model-q5_k.gguf",
            mmproj="models/llava/mmproj-model-f16.gguf",
            system_prompt="You are an image analysis expert. You give helpful and polite answers to questions. Your job is to describe the setting and context of an image, as concisely as possible.",
            user_prompt="Describe the setting and context of this image.",
            image=frame,
            temp=0.1,
            length_max_target=128,
        )
        video_settings.append(setting)
        logger.info(f'frame: {frame}')
    text_setting = '\n'.join(video_settings)
    token_count = get_token_count(str(text_setting)) + 60
    total_token_limit = int(1.8*token_count)
    # if longer context window is needed
    OPENAI_MODEL = "gpt-3.5-turbo-16k" if total_token_limit > MODEL_TOKEN_LIMIT else  "gpt-3.5-turbo"
    max_tokens = int(0.75*token_count)
    # max_tokens = int(0.5*MODEL_TOKEN_LIMIT_16K) if total_token_limit > MODEL_TOKEN_LIMIT else int(0.7*MODEL_TOKEN_LIMIT)
    logger.info(f'token in {token_count}; token total model limit {MODEL_TOKEN_LIMIT}; token total limit {total_token_limit}; token limit out {max_tokens}; model {OPENAI_MODEL}')
    text_setting = call_gpt_api_simple(client=openai_client, text=str(text_setting), user_prompt_supplemental=f"", model=OPENAI_MODEL, temperature=0.3, max_tokens=max_tokens) if text_setting else ''
    logger.info(f'gpt api-d + cleaned setting: {text_setting}')
    text_setting = text_setting if text_setting and coherence_score(text_setting) > 0.725 else ''

    # # TODO: figure out `UnboundLocalError: local variable 'OPENAI_MODEL' referenced before assignment`
    # # extra 220 tokens to account for system and user prompts
    # token_count = get_token_count(str(text_video)) + get_token_count(str(text_setting)) + 220
    # # if longer context window is needed
    # OPENAI_MODEL = "gpt-3.5-turbo-16k" if token_count > MODEL_TOKEN_LIMIT else  "gpt-3.5-turbo"
    # max_tokens = int(0.8*MODEL_TOKEN_LIMIT) if token_count > MODEL_TOKEN_LIMIT else int(0.8*MODEL_TOKEN_LIMIT_16K)
    # text_setting = call_gpt_api(
    #     client=openai_client,
    #     text=str(text_video),
    #     user_prompt_supplemental=f"The background setting and context of this video is: {text_setting}.",
    #     format="SETTING",
    #     model=OPENAI_MODEL,
    # ) if text_video else ''
    # logger.info(f'gpt api-d + cleaned setting: {text_setting}')
    # text_setting = text_setting if text_setting and coherence_score(text_setting) > 0.725 else ''

    # TODO (v1.5) (partially completed): add a "version" to upserted vectors, loosely corresponding to the quality of audio/video models used. so if the code is switche to a new model, you know whether to process existing videos again on the new models
    logger.info(f'EXTRACTING METADATA')
    text_metadata = extract_metadata()
    logger.info(f'extracted + cleaned metadata: {text_metadata}')
    
    # TODO: use some schema library to maintain this response format
    response_json = {
        'timestamp':str(datetime.now(timezone.utc)),
        'type':'reel',
        'id':video_id,
        'content':{
            'text_audio':text_audio,
            'text_video':text_video,
            'text_setting':text_setting,
            'text_metadata':text_metadata
        },
        'version':PROJECT_VERSION
    }
    logger.info(f'\nFINAL OUTPUT:\n{json.dumps(response_json)}')
    
    # save text data in mongo db
    insert_document(mdb_client, [response_json])

    # TODO (v1): if transcribed output already exists in ^ DB in current version, skip the whole training
    response_content = text_combine(text_audio, text_video, text_setting, text_metadata)
    
    # TODO (v1): fix upsert namespace to be platform.const
    logging.info(f"creating embeddings for content")
    embeddings = get_embedding(openai_client, response_content)

    logging.info(f'batching and upserting to pinecone')
    num_batches = int(len(embeddings)/250) + 1
    for i in range(num_batches):
        embeddings_batch = embeddings[i*250:(i+1)*250]
        upsert_response = pinecone_upsert(embeddings_batch, namespace=str(response_json['type']), id=str(response_json['id']))
        logging.info(f'pinecone upsert response: {upsert_response}')

    # TODO (v2): do some validation before returning 200, this could be dangerous
    return HTMLResponse(status_code=200)

    # TODO (v2): generate new metadata tags (subject, hashtags, etc) to be used as additional metadata (helpful for semantic search filters)

if __name__ == "__main__":

    uvicorn.run("main:app", port=5004, log_level="debug", reload=True)

    # argument_list = sys.argv[1:]
    # options = "v:"
    # long_options = ["video"]
    
    # try:
    #     # Parsing argument
    #     arguments, values = getopt.getopt(argument_list, options, long_options)
        
    #     # checking each argument
    #     for current_argument, current_value in arguments:
    #         if current_argument in ('-v', '--video'):
    #             logger.info(f'Downloading video {current_value}')
    #             train(current_value)
                
    # except getopt.error as err:
    #     print (str(err))



# # -----------------------ARCHIVE-----------------------



# ret, thresh_binary_otsu = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# thresh_adaptive_mean = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
# thresh_adaptive_gaussian = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 15)

# cv2.imshow('simple binary inv', thresh_simple_binary_inv)
# cv2.imshow('binary + otsu', thresh_binary_otsu)
# cv2.imshow('adaptive mean', thresh_adaptive_mean)
# cv2.imshow('adaptive gaussian', thresh_adaptive_gaussian)
# cv2.waitKey(0)



# # find contours
# contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# # create a copy of image
# f2 = frame.copy()

# # loop through the identified contours and crop rectangular part
# frame_list = []
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
    
#     # Drawing a rectangle on copied image
#     rect = cv2.rectangle(f2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     # Cropping the text block for giving input to OCR
#     cropped = f2[y:y + h, x:x + w]

#     # cv2.imshow('adaptive gaussian + rectangular kernel dilation + countouring', dilation)
#     # cv2.waitKey(0)