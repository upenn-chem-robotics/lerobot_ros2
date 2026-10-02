# Finish notification and functional corrections

- Removes remaining import-time writes into OpenCV's installed package directory from deploy and DAgger modules.
- Resolves `ffmpeg` and `ffprobe` from the activated Conda environment instead of assuming `/bin`.
- Sends a best-effort completion notification after the summary is written.

Notification lookup order:

1. `send-notify`, invoked as `send-notify TITLE MESSAGE`
2. `notify-send TITLE MESSAGE`
3. no notification when neither command exists

Disable notifications with:

```bash
NOTIFY_ON_FINISH=0 ./verify_public_release.sh
```

Notification failures never change the verification result.
