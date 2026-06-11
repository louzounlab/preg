# Deployment

The app ships as a single Docker image: Flask + Gunicorn on Python 3.11-slim,
with PyTorch installed from the CPU wheel index. All four model adapters
(twin-pe, twin-fwe, gdm, pepred) run in-process — no separate workers.

## Files added for deploy

- `Dockerfile` — production image (gunicorn, non-root user, healthcheck).
- `docker-compose.yml` — single-service stack, persistent volume for the
  per-request artifacts the twin-fwe adapter writes under `static/`.
- `.dockerignore` — keeps `.venv`, `.git`, `.env`, and the per-run static
  output out of the build context.
- `requirements.txt` — consolidated; previously each model subdir had its own.
- `.env.example` — full set of production env vars.

## 1. Prepare the host

Install Docker Engine + the Compose plugin. Anything 24.0+ is fine.

## 2. Configure secrets

```bash
cp .env.example .env
# edit .env: set FLASK_SECRET_KEY (long random string) and Google OAuth creds
```

In the Google Cloud console, add the production callback URL to your OAuth
client's **Authorized redirect URIs**:

```
https://<your-domain>/auth/callback
```

## 3. Build & run

```bash
docker compose build
docker compose up -d
docker compose logs -f web
```

The app listens on `:8000`. Put nginx / Caddy / a cloud load balancer in front
to terminate TLS and proxy to it. The app trusts `X-Forwarded-Proto` / `-Host`
/ `-For` by default (toggle with `TRUST_PROXY=0`), so `url_for(_external=True)`
returns `https://...` URLs and the OAuth redirect_uri matches what's
registered with Google.

### Minimal nginx snippet

```nginx
server {
  listen 443 ssl http2;
  server_name your-domain;
  # ... ssl_certificate, ssl_certificate_key ...

  location / {
    proxy_pass         http://127.0.0.1:8000;
    proxy_set_header   Host              $host;
    proxy_set_header   X-Real-IP         $remote_addr;
    proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_read_timeout 120s;
  }
}
```

## 4. Tuning knobs (env vars)

| Var                | Default      | Notes                                            |
| ------------------ | ------------ | ------------------------------------------------ |
| `PORT`             | `8000`       | Container port gunicorn binds.                   |
| `WEB_CONCURRENCY`  | `2`          | Gunicorn worker processes.                       |
| `GUNICORN_TIMEOUT` | `120`        | Seconds; bump if predictions are slow.           |
| `FLASK_ENV`        | `production` | Set to `development` to relax secure cookies.    |
| `TRUST_PROXY`      | `1`          | Set `0` if **not** behind a reverse proxy.       |

Each worker loads the LightGBM + PyTorch models on first request via
`lru_cache`, so memory scales roughly linearly with `WEB_CONCURRENCY`. Start
at 2 and scale up only if you have CPU + RAM headroom.

## 5. Persistent data

The twin-fwe adapter writes per-request plots and CSVs to
`static/<timestamp>/`. These are **ephemeral** — they live inside the
container and are lost on restart. We deliberately don't volume-mount
`/app/static` because that would shadow the baked-in CSS/JS/CSV assets the
image ships with. If you need long-term persistence of the generated runs,
the right fix is to write them to a dedicated path (e.g. `/var/runs/`) and
mount a volume there — not to overlay all of `static/`.

## 6. Updating

```bash
git pull
docker compose build
docker compose up -d
```

Old images: `docker image prune -f`.

## 7. Local dev (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
cp .env.example .env  # set FLASK_ENV=development
python app.py
```
