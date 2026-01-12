
from celery import Celery


celery_app = Celery(
    "distributed_training",
    broker = "redis://localhost:6379/0",
    backend = 'redis://localhost:6379/0',
    include=['tasks.train_task', 'tasks.aggregate_task']
)


celery_app.conf.update(
    task_serializer = 'json',
    accept_content = ['json'],
    result_serializer = 'json',
    timezone='Asia/Kolkata',
    enable_utc=False,
    task_track_started=True,
    task_time_limit=1800,
    task_ignore_result=False,
    task_annotations={
        '*': {'max_retries': 3, 'default_retry_delay': 60, 'autoretry_for': (Exception,)}
    }
)
    
