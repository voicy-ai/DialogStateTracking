from flask import Flask, render_template, request
from flask import jsonify

import main as botmodule


# global
bot = None
app = Flask(__name__,static_url_path="/static")


'''

 Routing
'''
# @GET
#  PATH : /query
@app.route('/query', methods=['GET'])
def reply():
    return bot.reply(request.args['msg'])

    '''
    return jsonify( {
        'ans' : 'dummy text'
        })

    '''

# render page "index.html"
#  PATH : /
@app.route("/")
def index():
    return render_template("index.html")


if (__name__ == "__main__"):
    # before starting the app, init model
    bot = botmodule.main(['--ui', '--task_id=1'])
    # start app
    app.run(port = 5000)
