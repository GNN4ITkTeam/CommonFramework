
df -h /dev/shm

ls -la

export WANDB_DISABLED=true

sed -i 's/accelerator: cpu/accelerator: cuda/g' *.yaml

pytest --cov --cov-report term --cov-report xml:coverage.xml

ls -la