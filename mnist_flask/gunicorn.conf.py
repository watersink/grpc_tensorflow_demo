bind = '0.0.0.0:5000'
workers = 2
threads = 1
max_requests = 1000
max_requests_jitter = 40

worker_connections = 1000
daemon = True
debug = False

pidfile = 'logs/gunicorn.pid'
accesslog = 'logs/gunicorn.access'
errorlog = 'logs/gunicorn.error'
loglevel = 'info'

