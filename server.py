from flask import Flask, render_template, request, json
from whoosher import *
app = Flask(__name__)


@app.route("/")
def hello():
    # return "Hello World!"
    return render_template("index.html")

@app.route("/upload", methods = ["POST"])
def uploadData():
   # return "Hello World! " + request.args["aunty"]
   queryText = request.args["aunty"]
   readFromDirectory()
   indexDir = createIndexFromCaptions(listOfCaptions)
   topResult = searchIndexForQuery(indexDir,queryText)
   return topResult


if __name__ == "__main__":
    app.run()
