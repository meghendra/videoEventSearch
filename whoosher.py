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
 ["a man is playing with a dog"]
,["a man is riding a motorcycle"]
,["a cat is playing"]
,["a man is talking"]
,["a man is taking a baby"]
,["a man is using a knife"]
,["a man is taking a face"]
,["two boys are dancing"]
,["a woman is holding a baby"]
,["a woman is plucking a face"]
,["a cat is playing with a cat"]
,["a woman is dancing"]
,["a girl is walking"]
,["people are playing"]
,["a man is pushing a car"]
,["a man is playing a card"]
,["a boy is running"]
,["a man is dancing"]
,["a man is riding a bike"]
,["two men are sitting on a table"]
,["a dog is playing"]
,["a man is slicing a potato"]
,["a man is playing a guitar"]
,["a cat is playing"]
,["a woman is driving a car"]
,["a man is driving a car"]
,["a dog is playing"]
,["a woman is mixing ingredients in a bowl"]
,["a tiger is chasing a deer"]
,["a man is talking"]
,["the man is doing the something"]
,["a boy is running"]
,["a man is taking a bag"]
,["a man is slicing a potato"]
,["a monkey is walking"]
,["a polar bear is walking"]
,["a man is walking on the beach"]
,["a man is playing a guitar"]
,["two women are playing"]
,["a cat is running"]
,["a man is shooting a gun"]
,["a man is slicing bread"]
,["a man is cutting a piece of bread"]
,["a man is putting some food"]
,["a man is making pizza"]
,["a man is driving a car"]
,["a man is playing with a cat"]
,["a person is cooking"]
,["a woman is cutting a apple"]
,["a man is cooking"]
,["two elephants are walking"]
,["the animal is eating"]
,["a man is playing a flute"]
,["a band is performing on stage"]
,["a woman is riding a horse"]
,["a man is playing a game"]
,["a man is driving a car"]
,["a man is playing soccer"]
,["a dog is playing"]
,["a man is riding a boat"]
,["a man is cooking"]
,["a man is shooting a gun"]]