#!/bin/bash
celery -A worker.celery_app worker --loglevel=info &
uvicorn backend:app --host 0.0.0.0 --port $PORT