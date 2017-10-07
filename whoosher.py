from whoosh.index import create_in
from whoosh.fields import *
from whoosh import qparser
import whoosh.index as index

def createIndexFromCaptions(listOfCaptions):
    dirName = "indexdir"
    schema = Schema(title=TEXT(stored=True), content=TEXT)
    # create index
    ix = create_in(dirName, schema)
    writer = ix.writer()
    for caption in listOfCaptions:
        writer.add_document(title=caption[0], content=caption[0])
    writer.commit()
    return dirName

def searchIndexForQuery(dirName,queryTerms):
    schema = Schema(title=TEXT(stored=True), content=TEXT)
    ix = index.open_dir(dirName)
    # search content for query term
    with ix.searcher() as searcher:
        parser = qparser.QueryParser("content", schema=ix.schema,group=qparser.OrGroup)
        query = parser.parse(queryTerms)
        results = searcher.search(query)
        if len(results) > 0:
            resultList = []
            for result in results:
                resultList.append(result["title"])
            return resultList[0]
        else:
            return ["middleFinger"][0]

l = [
 [u"a man is playing with a dog"]
,[u"a man is riding a motorcycle"]
,[u"a cat is playing"]
,[u"a man is talking"]
,[u"a man is taking a baby"]
,[u"a man is using a knife"]
,[u"a man is taking a face"]
,[u"two boys are dancing"]
,[u"a woman is holding a baby"]
,[u"a woman is plucking a face"]
,[u"a cat is playing with a cat"]
,[u"a woman is dancing"]
,[u"a girl is walking"]
,[u"people are playing"]
,[u"a man is pushing a car"]
,[u"a man is playing a card"]
,[u"a boy is running"]
,[u"a man is dancing"]
,[u"a man is riding a bike"]
,[u"two men are sitting on a table"]
,[u"a dog is playing"]
,[u"a man is slicing a potato"]
,[u"a man is playing a guitar"]
,[u"a cat is playing"]
,[u"a woman is driving a car"]
,[u"a man is driving a car"]
,[u"a dog is playing"]
,[u"a woman is mixing ingredients in a bowl"]
,[u"a tiger is chasing a deer"]
,[u"a man is talking"]
,[u"the man is doing the something"]
,[u"a boy is running"]
,[u"a man is taking a bag"]
,[u"a man is slicing a potato"]
,[u"a monkey is walking"]
,[u"a polar bear is walking"]
,[u"a man is walking on the beach"]
,[u"a man is playing a guitar"]
,[u"two women are playing"]
,[u"a cat is running"]
,[u"a man is shooting a gun"]
,[u"a man is slicing bread"]
,[u"a man is cutting a piece of bread"]
,[u"a man is putting some food"]
,[u"a man is making pizza"]
,[u"a man is driving a car"]
,[u"a man is playing with a cat"]
,[u"a person is cooking"]
,[u"a woman is cutting a apple"]
,[u"a man is cooking"]
,[u"two elephants are walking"]
,[u"the animal is eating"]
,[u"a man is playing a flute"]
,[u"a band is performing on stage"]
,[u"a woman is riding a horse"]
,[u"a man is playing a game"]
,[u"a man is driving a car"]
,[u"a man is playing soccer"]
,[u"a dog is playing"]
,[u"a man is riding a boat"]
,[u"a man is cooking"]
,[u"a man is shooting a gun"]]