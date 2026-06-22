import os

from authlib.integrations.flask_client import OAuth
from flask import Blueprint, current_app, redirect, session, url_for

import config

auth = Blueprint("auth", __name__)
oauth = OAuth()


def init_oauth(app):
    oauth.init_app(app)
    oauth.register(
        name="google",
        client_id=config.GOOGLE_CLIENT_ID,
        client_secret=config.GOOGLE_CLIENT_SECRET,
        server_metadata_url=config.GOOGLE_DISCOVERY_URL,
        client_kwargs={"scope": "openid email profile"},
    )


def _extract_userinfo(token):
    userinfo = token.get("userinfo") or {}
    if userinfo.get("sub"):
        return userinfo

    try:
        response = oauth.google.userinfo()
        if response is not None:
            fetched_userinfo = response.json() or {}
            if fetched_userinfo.get("sub"):
                return fetched_userinfo
    except Exception:
        current_app.logger.exception("Failed to fetch Google userinfo via Authlib helper")

    try:
        id_token_claims = oauth.google.parse_id_token(token, nonce=None)
        if id_token_claims:
            return dict(id_token_claims)
    except Exception:
        current_app.logger.exception("Failed to parse Google ID token")

    try:
        response = oauth.google.get("userinfo")
        if response is not None:
            fetched_userinfo = response.json() or {}
            if fetched_userinfo.get("sub"):
                return fetched_userinfo
    except Exception:
        current_app.logger.exception("Failed to fetch Google userinfo endpoint")

    return userinfo


@auth.route("/login")
def login():
    # Local development bypass: when FLASK_ENV=development (and Google OAuth is
    # not configured) sign in a fake user so the protected model pages are
    # reachable without setting up real OAuth credentials.
    if os.environ.get("FLASK_ENV") == "development" and not config.GOOGLE_CLIENT_ID:
        session["user"] = {
            "email": "dev@localhost",
            "name": "Local Dev",
            "picture": None,
        }
        next_url = session.pop("next", None)
        if not next_url or not next_url.startswith("/") or next_url[:2] in ("//", "/\\"):
            next_url = "/Home"
        return redirect(next_url)

    redirect_uri = url_for("auth.callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth.route("/auth/callback")
def callback():
    token = oauth.google.authorize_access_token()
    userinfo = _extract_userinfo(token)

    session["user"] = {
        "email": userinfo.get("email"),
        "name": userinfo.get("name") or userinfo.get("email"),
        "picture": userinfo.get("picture"),
    }
    next_url = session.pop("next", None)
    # Only follow same-site relative paths. Reject absolute URLs and
    # protocol-relative paths ("//evil.com") to avoid an open redirect.
    # Browsers may treat "/\" like "//", so reject both as the second char.
    if (
        not next_url
        or not next_url.startswith("/")
        or next_url[:2] in ("//", "/\\")
    ):
        next_url = "/Home"
    return redirect(next_url)


@auth.route("/logout")
def logout():
    session.clear()
    return redirect("/Home")
