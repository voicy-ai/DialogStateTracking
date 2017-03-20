from flask import Flask, render_template, request
from flask import jsonify


app = Flask(__name__,static_url_path="/static")


'''

 Routing
'''
# @POST
#  PATH : /message
@app.route('/query', methods=['POST'])
def reply():
    '''
    stories = request.form['stories']
    ques  = request.form['ques']

    '''
    return jsonify( {
        'ans' : 'dummy text'
        })


# render page "index.html"
#  PATH : /
@app.route("/")
def index():
    return render_template("index.html")



if (__name__ == "__main__"):
    # start app
    app.run(port = 5000)
