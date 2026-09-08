from functools import wraps

from flask import redirect, request, session


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            session["next"] = request.full_path if request.query_string else request.path
            # Land on the home page with the sign-in notice dialog open, so the
            # notice is shown before anything is transferred to Google.
            return redirect("/Home?signin=1")
        return view(*args, **kwargs)

    return wrapper
