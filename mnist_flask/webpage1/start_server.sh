lsof -i:5000|grep TCP|awk '{print$2}'|xargs kill -9
gunicorn server:app -c gunicorn.conf.py
