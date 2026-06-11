from functools import wraps

from flask import redirect, request, session, url_for


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            session["next"] = request.full_path if request.query_string else request.path
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)

    return wrapper
