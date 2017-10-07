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
   returnDict = { "url" : topResult[0], "time" : topResult[1]}
   return jsonify(returnDict)


if __name__ == "__main__":
    readFromDirectory()
    # print listOfCaptions
    createIndexFromCaptions(listOfCaptions)
    app.run()
