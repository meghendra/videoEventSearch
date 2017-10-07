from whoosh.index import create_in
from whoosh.fields import *
from whoosh import qparser
import whoosh.index as index
import os

listOfCaptions = []

def readFromDirectory():
    for filename in os.listdir("./captions"):
        readFromFile("./captions/"+filename)

def readFromFile(filename) :
    joda_string = ""
    with open(filename, "r") as f :
        line = f.readline()
        joda_string += line
    captionList = eval(joda_string)
    for s in captionList :
        listOfCaptions.append(s)

def createIndexFromCaptions(listOfCaptions):
    dirName = "indexdir"
    schema = Schema(title=TEXT(stored=True), time=TEXT(stored=True), content=TEXT)
    # create index
    ix = create_in(dirName, schema)
    writer = ix.writer()
    for caption in listOfCaptions:
        # print "@@ title={} time={} content={}".format(caption[0].split("/")[-1], caption[1], caption[2])
        writer.add_document(title=(caption[0].split("/")[-1].split(".")[0] + ".mp4").decode('utf-8'), time=str(caption[1]).decode('utf-8'), content=caption[2])
    writer.commit()
    return dirName

def addCaptionsToIndex(dirName,listOfCaptions):
    ix = index.open_dir(dirName)
    writer = ix.writer()
    for caption in listOfCaptions:
        writer.add_document(title=caption[0].split("/")[-1].decode('utf-8'), time=str(caption[1]).decode('utf-8'), content=caption[2])
    writer.commit()
    return True

def searchIndexForQuery(dirName,queryTerms):
    #schema = Schema(title=TEXT(stored=True), content=TEXT)
    ix = index.open_dir(dirName)
    # search content for query term
    with ix.searcher() as searcher:
        parser = qparser.QueryParser("content", schema=ix.schema,group=qparser.OrGroup)
        query = parser.parse(queryTerms)
        results = searcher.search(query)
        if len(results) > 0:
            resultList = []
            for result in results:
                resultList.append([result["title"],result["time"]])
            return resultList
        else:
            return ["middleFinger"][0]