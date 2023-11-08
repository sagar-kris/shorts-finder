import logging
import re
import numpy as np
from scipy.stats import kstest, powerlaw
import nltk
from nltk.corpus import words, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import brown
from nltk import pos_tag
from nltk.text import TextCollection
import openai
from fuzzywuzzy import fuzz

# TODO (v2): can these be removed or put into requirements.txt?
nltk.download('punkt')  # For tokenization
nltk.download('averaged_perceptron_tagger') # For POS tagging
nltk.download('words')  # For vocabulary matching
nltk.download('wordnet')
nltk.download('brown')

logger = logging.getLogger(__name__)

VOCABULARY_MATCHING_WEIGHT = 0.45
NGRAM_ANALYSIS_WEIGHT = 0.05
SENTENCE_STRUCTURE_WEIGHT = 0.45
TF_IDF_WEIGHT = 0.05

ENCODING_NAME = "cl100k_base"

class Platform:
    def __init__(self, const, stub, url) -> None:
        self.const = const
        self.stub = stub
        self.url = url

TP, RP, YP = Platform("TIKTOK", [], "https://www.tiktok.com"), Platform("REELS", ["reels", "reel"], "https://www.instagram.com"), Platform("YTSHORTS", ["shorts"], "https://www.youtube.com")

def find_substrings(main_string, substrings):
    # Compile all the substrings into a single regex pattern using the `join` method and the `|` operator
    pattern = '|'.join(re.escape(sub) for sub in substrings)
    
    # Use `search` to find the first occurrence of any substring in the main string
    match = re.search(pattern, main_string)
    
    # Return the found substring or None if no substring is found
    return match.group(0) if match else None

def clean_video_link(video_links: list[str]):
    video_links = [video_id.strip().strip('/') for video_id in video_links]
    videos_list = []
    for video_link in video_links:
        # determine platform
        platform_string = find_substrings(video_link, TP.stub+YP.stub+RP.stub)

        match platform_string:
            case "reels":
                platform = RP
                video_id = video_link[video_link.rindex('/reels')+7:]
            case "reel":
                platform = RP
                video_id = video_link[video_link.rindex('/reel')+6:]
            case "p":
                platform = RP
                video_id = video_link[video_link.rindex('/p')+3:]
            case "shorts":
                platform = YP
                video_id = video_link[video_link.rindex('/shorts')+8:]
            case default:
                platform = None
                video_id = None
        videos_list.append((platform, video_id))
    return videos_list

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = np.arange(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances

    return distances[-1]

def classify_distribution(data_struct):
    # extract frequency distribution
    data = [freq for (_, freq) in data_struct]

    # Compute Descriptive Statistics
    mean_val = np.mean(data)
    std_dev = np.std(data)
    
    # Use the parameters for the powerlaw fit to get a reference CDF
    params = powerlaw.fit(data)
    cdf_powerlaw = lambda x: powerlaw.cdf(x, *params)
    
    # Kolmogorov-Smirnov Test against the fitted power-law distribution
    _, p_value = kstest(data, cdf_powerlaw)
    
    # Classification Logic
    # Adjust the thresholds based on your data and specific requirements
    if mean_val < 4 and std_dev < 2 and p_value < 0.05:
        return 'FLAT_DISTRIBUTION'
    else:
        return 'POWERLAW_DISTRIBUTION'

def iqr_threshold(data_struct, multiplier=1):
    # extract the frequencies
    frequencies = [freq for (_, freq) in data_struct]

    # Computing the interquartile range (IQR) for the frequencies
    q1 = np.percentile(frequencies, 25)
    q3 = np.percentile(frequencies, 75)
    iqr = q3 - q1

    # Setting the threshold
    threshold = q1 + multiplier * iqr
    logger.info(f'iqr threshold: {threshold}\nq1: {q1}\nq3: {q3}\niqr: {iqr}\n')

    # Filtering the phrases based on the threshold
    filtered_phrases = [(phrase, freq) for (phrase, freq) in data_struct if freq >= threshold]

    return filtered_phrases

def text_cleanup(strings):
    """Find the useful strings in each subsequence with an edge case for the first string."""
    
    # Strip all empty strings
    strings = [string for string in strings if string]

    if len(strings) < 2:
        return strings

    # find the sharp dropoffs created by dissimilarity in the string sequence
    result_list = []
    prev_string = strings[0]
    redundance_count = 0
    
    for curr_string in strings[1:]:
        if fuzz.token_set_ratio(prev_string, curr_string) < 75:
            result_list.append((prev_string, redundance_count))
            redundance_count = 0
        
        # Update previous string
        prev_string = curr_string
        redundance_count += 1

    # add the last string
    result_list.append((prev_string, redundance_count))

    # some disjoints could be accidental, so try to re-merge similar groups
    result_list_final = result_list
    num_unifications = 1

    # keep doing pass throughs until no more merges possible
    while num_unifications and len(result_list_final) > 1:
        result_list_temp = []
        num_unifications = 0
        prev_string_tuple = result_list_final[0]
        index = 1

        # single pass through result_list_final
        while index < len(result_list_final):
            curr_string_tuple = result_list_final[index]
            # highly similar but disjointed groups, we want to merge
            if fuzz.token_set_ratio(prev_string_tuple[0], curr_string_tuple[0]) > 80:
                result_list_temp.append((prev_string_tuple[0], int(prev_string_tuple[1]+curr_string_tuple[1])))
                num_unifications += 1
                index += 1
                curr_string_tuple = result_list_final[index]
            # leave as is
            else:
                result_list_temp.append(prev_string_tuple)

            prev_string_tuple = curr_string_tuple
            index += 1    

        # add last item in list
        result_list_temp.append(prev_string_tuple)
        result_list_final = result_list_temp

    # classify the distribution and filter accordingly
    dist_type = classify_distribution(result_list_final)
    logger.info(f'dist_type: {dist_type}')
    if dist_type == 'FLAT_DISTRIBUTION':
        result_list_final = iqr_threshold(result_list_final) if dist_type == 'FLAT_DISTRIBUTION' else iqr_threshold(result_list_final, 3)

    return result_list_final

def vocabulary_matching_score(text, tokens):
    english_words = set(words.words())
    vocabulary_count = 0
    for token in tokens:
        token = token.lower()
        if token in english_words or wn.synsets(token):
            vocabulary_count += 1
    return vocabulary_count / len(tokens)

def ngram_analysis_score(text, n=2):
    # Create a list of n-grams from the Brown corpus
    brown_ngrams = list(ngrams(brown.words(), n))
    text_ngrams = list(ngrams(word_tokenize(text), n))
    
    common_ngrams = sum(1 for ng in text_ngrams if ng in brown_ngrams)
    return common_ngrams / len(text_ngrams)

def sentence_structure_score(text, tokens):
    tags = pos_tag(tokens)
    
    # Basic check: does the text have nouns and verbs?
    has_noun = any(tag.startswith('NN') for word, tag in tags)
    has_verb = any(tag.startswith('VB') for word, tag in tags)
    
    return float(has_noun and has_verb)

def tf_idf_score(text):
    # Use Brown words + text as corpus
    corpus = TextCollection([text] + [' '.join(brown.words())])
    tokens = word_tokenize(text)
    
    # Calculate the TF-IDF score for each word in the text
    tfidf_scores = [corpus.tf_idf(token, text) for token in tokens]
    
    # Normalize
    return sum(tfidf_scores) / len(tokens)

# TODO: coherence score doesn't really work, make it good; it needs to be able to decide if the audio or video source is clearly better based on the "type" of reel
def coherence_score(text):
    tokens = word_tokenize(text)

    v_score = vocabulary_matching_score(text, tokens) * VOCABULARY_MATCHING_WEIGHT
    n_score = ngram_analysis_score(text) * NGRAM_ANALYSIS_WEIGHT
    s_score = sentence_structure_score(text, tokens) * SENTENCE_STRUCTURE_WEIGHT
    t_score = tf_idf_score(text) * TF_IDF_WEIGHT

    logger.info(f'vocabulary matching score {v_score / VOCABULARY_MATCHING_WEIGHT}')
    logger.info(f'ngram analysis score {n_score / NGRAM_ANALYSIS_WEIGHT}')
    logger.info(f'sentence structure score {s_score / SENTENCE_STRUCTURE_WEIGHT}')
    logger.info(f'tf idf score {t_score / TF_IDF_WEIGHT}')
    
    c_score = v_score + n_score + s_score + t_score
    logger.info(f'coherence score: {c_score}')
    return c_score

def call_gpt_api_simple(
        client,
        text: str,
        user_prompt_supplemental: str="",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=4096,
) -> str:
    logger.info(f'Querying GPT API with model: {model} and temperature {temperature}')
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content" : f"You are a helpful english language assistant, good at consolidating and summarizing text."
            },
            {
                "role": "user",
                "content" : f"Your task is to consolidate the provided text into a single understandable paragraph, using the english phrases and sentences from the provided text. {user_prompt_supplemental}\nHere is the provided text for you to consolidate: {text}"
            }
        ]
    )
    logger.info(f'gpt api raw response: {response}')
    return response.choices[0].message.content

def call_gpt_api(
        client,
        text: str,
        user_prompt_supplemental: str="",
        format="AUDIO",
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=4096,
    ) -> str:
    if format == "AUDIO":
        data_format = f"text. This text is a paragraph or transcribed audio"
        user_prompt_supplemental = f""
    elif format == "VIDEO":
        data_format = f"list of string, weight pairs. The strings are phrases, possibly garbled and misspelled, and with unwanted characters interspersed throughout. The weight number next to each string is the importance of that phrase in your final output"
    elif format == "SETTING":
        data_format = f"text. This text is a paragraph or transcribed audio"
    else:
        logger.error(f"ERROR: Must be one of the available text formats: AUDIO or VIDEO")

    logger.info(f'Querying GPT API with model: {model} and temperature {temperature}')
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content" : f"You are an english languge expert, with experience extracting coherent sentences from repetitive, haphazard, obfuscated text. Your goal is to clean up text. Extract the english phrases and sentences from the provided text and combine it to form an understandable paragraph. You do not summarize, paraphrase, or discard any useful words or phrases. Salvage the legible english phrases and make sense of this text without hallucinating. Do not include an explanation, disclaimers, or notes in your output. Take a deep breath, you got this."
            },
            {
                "role": "user",
                "content" : f"Your task is to clean up the provided text into an understandable paragraph, using the english phrases and sentences from the provided text. The text is in the format of {data_format}. {user_prompt_supplemental}\nHere is the provided text for you to clean up: {text}"
            }
        ]
    )
    logger.info(f'gpt api raw response: {response}')
    return response.choices[0].message.content

# combine the different text extractions into one paragraph
def text_combine(a, b, c, d):
    # TODO (v1): figure out how to pass & unpack an undefined number of args
    return f'{a}\n{b}\n{c}\n{d}'


#---------------ARCHIVE----------------
    # in for loop of text_cleanup():
        # max_length = max(len(prev_string), len(curr_string))
        
        # # Calculate length difference
        # length_diff_threshold = round(0.4 * max_length, 1)
        # length_diff = len(prev_string) - len(curr_string)
        
        # # Calculate string distance and threshold
        # distance_threshold = round(0.8 * max_length, 1)
        # distance = levenshtein_distance(prev_string, curr_string)

        # # Check if we need to consider previous string as the end of a subsequence
        # logger.info(prev_string, length_diff_threshold, length_diff, distance_threshold, distance)

        # # either one is above threshold
        # if length_diff > length_diff_threshold or distance > distance_threshold:
        #     result_list.append((prev_string, redundance_count))
        #     redundance_count = 0
        # # both are just below threshold
        # elif (length_diff + distance) > (0.85 * (length_diff_threshold + distance_threshold)):
        #     result_list.append((prev_string, redundance_count))
        #     redundance_count = 0
