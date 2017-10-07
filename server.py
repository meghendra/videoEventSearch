from flask import Flask, render_template, request, json
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
   readFromDirectory()
   # print listOfCaptions
   indexDir = createIndexFromCaptions(listOfCaptions)
   topResult = searchIndexForQuery(indexDir,queryText)
   return topResult[0]


if __name__ == "__main__":
    app.run()
