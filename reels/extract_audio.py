# from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
# from IPython.display import Audio
import logging
import os
import re
from huggingsound import SpeechRecognitionModel
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
import moviepy.editor as mp

logger = logging.getLogger(__name__)

# extract audio
def extract_audio(id, path, start, end, dest_path):
  clip = os.path.join(path, id)
  clip = mp.VideoFileClip(clip) # most recent clip
  end = min(clip.duration, end)

  # Save the paths for later
  dest_path = os.path.join(dest_path, id)

  # Extract Audio-only from mp4
  # TODO: make sure no audio segments are being skipped
  if not os.path.exists(dest_path):
    os.makedirs(dest_path)
    for i in range(start, int(end), 10):
      sub_end = min(i+10, end)
      sub_clip = clip.subclip(i, sub_end)

      sub_clip.audio.write_audiofile(f'{dest_path}/audio_{str(i)}.mp3')

    # # play a section of the audio
    # Audio(audio_clip_paths[0])

  model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")

  audio_clip_paths = [os.path.join(dest_path, file) for file in os.listdir(dest_path)]
  audio_clip_paths = sorted(audio_clip_paths, key=lambda x: int(re.search(r'audio_(\d+)\.mp3', x).group(1)))
  transcriptions = model.transcribe(audio_clip_paths)
  transcription = [transcriptions[i]['transcription'] for i in range(len(transcriptions))]
  transcription = ' '.join(transcription)
  return transcription