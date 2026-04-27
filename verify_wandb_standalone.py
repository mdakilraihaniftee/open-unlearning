import os
import wandb


def load_env():
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        print(f"Loading .env from {env_path}")
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
    else:
        print(".env not found")


load_env()

print(f"WANDB_PROJECT: {os.environ.get('WANDB_PROJECT')}")
print(f"WANDB_API_KEY exists: {bool(os.environ.get('WANDB_API_KEY'))}")

try:
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    run = wandb.init(project=os.environ.get("WANDB_PROJECT"), name="verify_antigravity")
    print(f"WandB run initialized: {run.url}")
    run.finish()
    print("WandB verification successful")
except Exception as e:
    print(f"WandB verification failed: {e}")
