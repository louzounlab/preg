import os

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

import config
from api.predict_routes import api
from views.auth_routes import auth, init_oauth
from views.ui_routes import ui

app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY

# Behind an HTTPS-terminating reverse proxy (nginx, traefik, a load balancer),
# trust X-Forwarded-* so that url_for(_external=True) generates https:// URLs
# and the OAuth redirect_uri matches the one registered with Google.
if os.environ.get("TRUST_PROXY", "1") == "1":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Production cookie hardening (opt out by setting FLASK_ENV=development).
if os.environ.get("FLASK_ENV", "production") != "development":
    app.config.update(
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PREFERRED_URL_SCHEME="https",
    )

init_oauth(app)

app.register_blueprint(ui)
app.register_blueprint(api)
app.register_blueprint(auth)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
