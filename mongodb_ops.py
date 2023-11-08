import os, sys
import json
import logging
import pymongo

logger = logging.getLogger(__name__)

# create a new client and connect to the server
def create_client():
    username_mdb, password_mdb = os.getenv("MONGODB_USERNAME"), os.getenv("MONGODB_PASSWORD")
    uri = f"mongodb+srv://{username_mdb}:{password_mdb}@reels-to-text.tb1inhg.mongodb.net/?retryWrites=true&w=majority"
    client = pymongo.mongo_client.MongoClient(uri, server_api=pymongo.server_api.ServerApi('1'))
    return client

# send a ping to confirm a successful connection
def test_client(client):
    try:
        client.admin.command('ping')
        logger.info("successfully connected to mongoDB and pinged deployment")
    except Exception as e:
        logger.error(e)

# return the db collection
def get_collection(client):
    db = client.test
    collection = db["yt-shorts"]
    return collection

# insert a document into a collection
def insert_document(client, document):
    collection = get_collection(client)
    try:
        result = collection.insert_many(document)
    except pymongo.errors.OperationFailure:
        logger.error("authentication error received; database user may not be authorized to perform write operations")
    else:
        logger.info(f"{len(result.inserted_ids)} documents inserted successfully")
    return


# # test document
# shorts = [{"timestamp": "2023-11-07 20:18:59.041907+00:00", 
#            "type": "reel", 
#            "id": "YTSHORTS_6EcqX1OXuig.mp4",
#            "content": {
#                "text_audio": "The text is a paragraph or transcribed audio. I anticipated the reform version deemed by the political opponents. The president is a prominent figure in the community.", 
#                "text_video": "Advanced bots play all day. They could make the game play for you. Very useful.", 
#                "text_setting": "the image, a man with dreadlocks can be seen standing in a gym, wearing black pants and flexing his muscles. There are two benches in the background, one on the left side and another further back, along with a sports ball near the right edge. The man's pose suggests that he may be showcasing his fitness or engaging in a workout session. The gym environment provides an ideal space for such activities, offering various equipment and facilities. The scene is set against a white wall, providing a clean and minimalistic backdrop. This image emphasizes the importance of maintaining a healthy lifestyle through regular exercise and dedication to personal growth.", 
#                "text_metadata": ""
#             }
#         }]

# # to drop a document
# try:
#   my_collection.drop()

# # return a friendly error if an authentication error is thrown
# except pymongo.errors.OperationFailure:
#   logger.info("An authentication error was received. Are your username and password correct in your connection string?")
#   sys.exit(1)
