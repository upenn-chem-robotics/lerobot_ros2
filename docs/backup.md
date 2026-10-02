# Hugging Face backup

Copy `examples/hf-backup.yaml` into ignored local configuration, keep repositories private unless reviewed, and authenticate through the mounted cache or an injected secret.

```bash
cp examples/hf-backup.yaml config.local/hf-backup.yaml
docker compose run --rm tools lerobot-ros-backup --config /config/hf-backup.yaml --help
```

Never commit tokens, private inventories, operator names, or participant data. Replacing current files does not erase Git history.
