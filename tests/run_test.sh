
df -h /dev/shm

ls -la

export WANDB_DISABLED=true

pytest --cov --cov-report term --cov-report xml:coverage.xml

ls -la