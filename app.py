from flask import Flask
from views.ui_routes import ui
from api.predict_routes import api

app = Flask(__name__)
#register the ui routes blueprint
app.register_blueprint(ui)
app.register_blueprint(api) # predictions models 

if __name__ == '__main__':
    app.run(debug=True)