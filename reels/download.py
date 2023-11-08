import logging
import os
import shutil
from reels.text_operations import Platform

logger = logging.getLogger(__name__)

# find the downloaded mp4 files in root directory
def move_files(type, source_dir, dest_dir, id, const="YTSHORTS"):
    file_name_new = ''
    logging.info(f'dest_dir {dest_dir}')
    for file_name in os.listdir(source_dir):
        if file_name.endswith(type):
            # extract the filename, rename it to "[platform.stub]_[video_id].mp4", and move it to right dir
            file_path_old = os.path.join(source_dir, file_name)
            logging.info(f'file_path_old {file_path_old}')
            file_name_new = f'{const}_{id}{type}'
            file_path_new = os.path.join(dest_dir, file_name_new)
            logging.info(f'file_path_new {file_path_new}')
            shutil.move(file_path_old, file_path_new)
    return file_name_new

# Download the clip as mp4 & rename it for usability
def download_clip(platform: Platform, id: str, source: str, dest: str):
    for stub in platform.stub:
        match platform.const:
            case "REELS":
                try:
                    logging.info(f'downloading clip {platform.url}/{stub}/{id}')
                    os.system('yt-dlp -vU {0} --cookies ./reels/instagram_cookies_4.txt --recode-video mp4'.format(f'{platform.url}/{stub}/{id}/'))
                except:
                    pass
            case "YTSHORTS":
                try:
                    logging.info(f'downloading clip {platform.url}/{stub}/{id}')
                    # TODO:  add `--write-info-json` flag to extract metadata, then delete file
                    os.system('yt-dlp -vU {0} --recode-video mp4'.format(f'{platform.url}/{stub}/{id}/'))
                except:
                    pass
            case default:
                logging.info(f'no case match for downloading clip {platform.url}/{stub}/{id}')
                continue
        # TODO: move .info.json files to videos/metadata
        return move_files('.mp4', source, dest, id, platform.const)
        

'''
def download_clip(id, type='reels'):
    logging.info(f'downloading clip {id}')
    os.system('yt-dlp {0} --cookies ./reels/instagram_cookies_3.txt --recode-video mp4'.format(f'https://www.instagram.com/{type}/{id}/'))

os.system('yt-dlp -v -j --print filename {0} --cookies ./reels/instagram_cookies.txt --recode-video mp4 --write-info-json --output "./videos/%(channel)s"'.format(f'https://www.instagram.com/{type}/{id}'))
'''

# # find the downloaded mp4 files in root directory
# def move_files(type, source_dir, dest_dir, id, stub="YTSHORTS"):
#     mp4_files = []
#     for file_name in os.listdir(source_dir):
#         if file_name.endswith('.mp4'):
#             # extract the filename, rename it to "[user]_[reel_id].mp4", and move it to right dir
#             file_name_new = f'{stub}_{id}'
#             file_name_old = os.path.basename(file)
#             file_name = file_name_old.split()
#             # file_name = file_name[2] + '_' + file_name[3][1:-5] + '.mp4'  # rename
#             file_name = file_name[3][1:-5] + '.mp4'  # rename
#             dest_path = f'{dest_dir}/{file_name}'
#             cmd = f'mv "{file_name_old}" {dest_path}' # move to new location
#             os.system(cmd)
#     return
