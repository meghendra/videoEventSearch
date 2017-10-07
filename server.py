from flask import Flask, render_template, request, json
app = Flask(__name__)

 
@app.route("/")
def hello():
    # return "Hello World!"
    return render_template("index.html")
 
@app.route("/upload", methods = ["POST"])
def uploadData():
    print request.args["aunty"]
    return "Hello World! " + request.args["aunty"]

if __name__ == "__main__":
    app.run()