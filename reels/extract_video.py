import logging
import os
import cv2
import pytesseract as tess

logger = logging.getLogger(__name__)

VALID_TWO_LETTER_WORDS_EN = ['aa', 'ab', 'ad', 'ae', 'ag', 'ah', 'ai', 'al', 'am', 'an', 'ar', 'as', 'at', 'aw', 'ax', 'ay', 'ba', 'be', 'bi', 'bo', 'by', 'da', 'de', 'do', 'ed', 'ef', 'eh', 'el', 'em', 'en', 'er', 'es', 'et', 'ex', 'fa', 'go', 'ha', 'he', 'hi', 'ho', 'id', 'if', 'in', 'is', 'it', 'jo', 'ka', 'ki', 'la', 'li', 'lo', 'ma', 'me', 'mi', 'mo', 'mu', 'my', 'na', 'ne', 'no', 'nu', 'od', 'oe', 'of', 'oh', 'oi', 'ok', 'om', 'on', 'op', 'or', 'os', 'ow', 'ox', 'oy', 'pa', 'pe', 'pi', 'qi', 're', 'sh', 'si', 'so', 'ta', 'ti', 'to', 'uh', 'um', 'un', 'up', 'us', 'ut', 'we', 'wo', 'xi', 'xu', 'ya', 'ye', 'yo', 'za', 'ch', 'di', 'ea', 'ee', 'gi', 'ko', 'ky', 'ny', 'ob', 'ou', 'po', 'st', 'te', 'ug', 'ur', 'yu', 'za', 'zo']

# clean up text
def cleanup_text(text: str):
    text = text.replace('\n', '')
    text = ''.join(char for char in text if char.isalpha() or char == ' ')
    filtered_words = [word for word in text.split() if len(word) > 2 or word in VALID_TWO_LETTER_WORDS_EN]
    text = ' '.join(filtered_words)
    return text

# TODO: break up this function further
def extract_video(video_name, videos_path, frames_path, frame_dropout,):
    video = os.path.join(videos_path, video_name)
    # start openCV video capture
    video = cv2.VideoCapture(video)

    # get frames from video (one time operation)
    frames_path = os.path.join(frames_path, video_name)
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

        index = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            frame_name = f'{frames_path}/frame_{str(index)}.png'
            if index % (frame_dropout ** frame_dropout) == 0:
                logging.info(f'Extracting frame {index} from video')
            
            # grayscale and threshold
            f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(frame_name, f)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            # don't need every frame, text is slow
            index += frame_dropout
            video.set(cv2.CAP_PROP_POS_FRAMES, index)

        # teardown
        video.release()
        cv2.destroyAllWindows()

    # extract text from each frame
    frame_dir = sorted(os.listdir(frames_path), key=lambda x: int(x[6:-4]))

    # TODO (v2): experiment with config='--psm 11'
    # use config='--psm 6'
    text_psm6 = []
    for index, frame_filename in enumerate(frame_dir):
        # if index%3 != 0:    # simply because the operation is $$, replace later
        #     continue
        if index % (frame_dropout ** 2) == 0:
            logging.info(f'Extracting text using psm6 from {frame_filename}')
        # logger.debug(text_psm6)
        frame_name = os.path.join(frames_path, frame_filename)
        frame = cv2.imread(frame_name)

        ret, thresh_simple_binary_inv = cv2.threshold(frame, 245, 255, cv2.THRESH_BINARY_INV)

        # specify structure shape and kernel size for text block identification
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # apply dilation on the threshold image
        dilation = cv2.dilate(thresh_simple_binary_inv, rect_kernel, iterations = 1)
        # cv2.imshow('simple binary inv + rectangular kernel dilation', dilation)
        # cv2.waitKey(0)
            
        # apply OCR on the cropped image
        frame_text_raw = tess.image_to_string(dilation, config=f'--psm 6')

        # clean up text
        frame_text = cleanup_text(text=frame_text_raw)

        # # add to list of extracted text
        text_psm6.append(frame_text)
    
    return text_psm6