lsof -i:5000|grep TCP|awk '{print$2}'|xargs kill -9
redis-server &
celery worker -A app.celery --loglevel=info &
gunicorn server:app -c gunicorn.conf.py
