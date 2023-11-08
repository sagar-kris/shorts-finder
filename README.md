# Shorts Finder

Unlock the power of multimodal video search, in your pocket. Shorts Finder searches over your personal collection of Youtube Shorts videos and surfaces only the most relevant results.

## Click to watch the Demo Video:
[![Shorts Finder - Demo Video] (https://img.youtube.com/vi/lSzqGAyStS8/maxresdefault.jpg)](https://www.youtube.com/shorts/lSzqGAyStS8)

-----------------------------------------------------------------------------------------------------------------------------------------------------------

## Training

Every time you come across a Short you like, simple copy the link and hit "Add to Collection". This step gathers metadata, transcribes the audio, performs computer vision on any text in the video, and uses AI-generated image captioning on select frames. Once all this information is available as text, it is pushed to both a NoSQL database as well as a vector database. Once the indexing is complete, your Short is ready to be searched.

## Inference

Type your search query and hit "Search my Shorts". The results should be embedded directly in the webpage, with the most relevant results first.


<!-- ## Architecture

## To get it running locally -->
