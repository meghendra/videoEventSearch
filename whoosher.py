from whoosh.index import create_in
from whoosh.fields import *
from whoosh import qparser
import whoosh.index as index

def createIndexFromCaptions(listOfCaptions):
    dirName = "indexdir"
    schema = Schema(title=TEXT(stored=True), time=TEXT(stored=True), content=TEXT)
    # create index
    ix = create_in(dirName, schema)
    writer = ix.writer()
    for caption in listOfCaptions:
        writer.add_document(title=caption[0], time=caption[1], content=caption[2])
    writer.commit()
    return dirName

def addCaptionsToIndex(dirName,listOfCaptions):
    ix = index.open_dir(dirName)
    writer = ix.writer()
    for caption in listOfCaptions:
        writer.add_document(title=caption[0], time=caption[1], content=caption[2])
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
            return resultList[0]
        else:
            return ["middleFinger"][0]

l = [
 [u"vid1",u"time1",u"a man is playing with a dog"]
,[u"vid4",u"time1",u"a man is riding a motorcycle"]
,[u"vid3",u"time1",u"a cat is playing"]
,[u"vid2",u"time1",u"a man is talking"]
,[u"vid6",u"time1",u"a man is taking a baby"]
,[u"vid9",u"time1",u"a man is using a knife"]
,[u"vid8",u"time1",u"a man is taking a face"]
,[u"vid10",u"time1",u"two boys are dancing"]
,[u"vid11",u"time1",u"a woman is holding a baby"]
,[u"vid12",u"time1",u"a woman is plucking a face"]
,[u"vid13",u"time1",u"a cat is playing with a cat"]
,[u"vid14",u"time1",u"a woman is dancing"]
,[u"vid15",u"time1",u"a girl is walking"]
,[u"vid16",u"time1",u"people are playing"]
,[u"vid17",u"time1",u"a man is pushing a car"]
,[u"vid18",u"time1",u"a man is playing a card"]
,[u"vid19",u"time1",u"a boy is running"]
,[u"vid20",u"time1",u"a man is dancing"]
,[u"vid21",u"time1",u"a man is riding a bike"]
,[u"vid24",u"time1",u"two men are sitting on a table"]
,[u"vid25",u"time1",u"a dog is playing"]
,[u"vid26",u"time1",u"a man is slicing a potato"]
,[u"vid29",u"time1",u"a man is playing a guitar"]
,[u"vid30",u"time1",u"a cat is playing"]
,[u"vid32",u"time1",u"a woman is driving a car"]
,[u"vid31",u"time1",u"a man is driving a car"]
,[u"vid34",u"time1",u"a dog is playing"]
]