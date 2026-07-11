import multiprocessing

bind = "0.0.0.0:8000"
workers = max(2, multiprocessing.cpu_count() * 2 + 1)
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
accesslog = "-"
errorlog = "-"
preload_app = False
