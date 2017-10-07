from flask import Flask, render_template, request, jsonify
from whoosher import *
app = Flask(__name__)


@app.route("/")
def hello():
    # return "Hello World!"
    return render_template("new_index.html")

@app.route("/upload", methods = ["POST"])
def uploadData():
   # return "Hello World! " + request.args["aunty"]
   queryText = request.args["aunty"]
   topResult = searchIndexForQuery("indexdir",queryText)
   if(False) > 1: 
        returnDict = { "first": {"url" : topResult[0][0], "time" : topResult[0][1]}, "second": {"url" : topResult[1][0], "time" : topResult[1][1]}}
   else :
        returnDict = { "first": {"url" : topResult[0][0], "time" : topResult[0][1]}}
   return jsonify(returnDict)


@app.route("/vid/<vid_id>")
def videoReturn(vid_id):
    return app.send_static_file(vid_id)


if __name__ == "__main__":
    readFromDirectory()
    # print listOfCaptions
    createIndexFromCaptions(listOfCaptions)
    app.run(threaded=True)
