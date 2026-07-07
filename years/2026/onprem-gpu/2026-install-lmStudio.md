For a headless Ubuntu server (no GUI), use **LM Studio CLI** (`lms`) and run the server in the background. This guide shows how to:

1. Install `lms` (CLI) on Ubuntu server  
2. Download and load a model  
3. Start the OpenAI-compatible server bound to all interfaces  
4. Access it from Windows clients  

***

## 1. Install LM Studio CLI (`lms`) on Ubuntu Server

### 1.1. Install required dependencies

```bash
sudo apt update
sudo apt install -y \
    npm \
    fuse \
    fuse3 \
    libfuse2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libx11-6 \
    libnss3 \
    libasound2 \
    libcups2 \
    xauth \
    xvfb \
    xfce4 \
    xfce4-goodies
```

These are needed for `lms` and the AppImage backend. [gistpad](https://gistpad.com/guide-installing-and-running-lm-studio-on-ubuntu-as-a-headless-server)

***

### 1.2. Download LM Studio AppImage (for the backend)

```bash
cd ~
wget https://installers.lmstudio.ai/linux/x64/0.3.6-8/LM-Studio-0.3.6-8-x64.AppImage \
    -O LM-Studio.AppImage

chmod +x LM-Studio.AppImage
```

You can also use the latest link from the LM Studio site if you prefer. [gistpad](https://gistpad.com/download/guide-installing-and-running-lm-studio-on-ubuntu-as-a-headless-server)

***

### 1.3. Run AppImage once (to initialize the backend)

Run it with a virtual display:

```bash
xvfb-run ./LM-Studio.AppImage --no-sandbox
```

This will create the necessary files under `~/.lmstudio`. You can immediately close it (Ctrl+C) after it starts. [github](https://github.com/lmstudio-ai/lms)

***

### 1.4. Install the `lms` CLI

```bash
# Install CLI via npm
sudo npm install -g lmstudio

# Bootstrap (creates ~/.lmstudio/bin/lms)
~/.lmstudio/bin/lms bootstrap
```

If `npm install -g lmstudio` doesn't give you `lms`, you can also use:

```bash
npx lmstudio install-cli
```

Confirm with `Yes` when prompted about PATH updates. [github](https://github.com/lmstudio-ai/lms)

Add `lms` to your PATH if not already:

```bash
export PATH="$PATH:/root/.lmstudio/bin"   # or your user path, e.g. /home/username/.lmstudio/bin
source ~/.profile
```

Verify:

```bash
lms --help
```

***

## 2. Start the LM Studio Daemon (headless engine)

Start the background daemon:

```bash
lms_daemon_start
```

Expected output:

```text
✔ LM Studio daemon started in the background.
```

Check status:

```bash
lms status
```

***

## 3. Download and Load a Model

### 3.1. (Optional) Download a small model first

Example: Qwen3 7B Instruct (GGUF, Q4_K_M):

```bash
lms get https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF@Q4_K_XL
```

Or for a smaller model:

```bash
lms get https://huggingface.co/lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF@Q4_K_M
```

***

### 3.2. Import the model (if needed)

If you downloaded a `.gguf` file manually:

```bash
lms import /path/to/model.gguf \
    --symbolic-link \
    --user-repo your-namespace/model-name
```

Then list available models:

```bash
lms models list
```

***

### 3.3. Load the model into the server

```bash
lms load Qwen3-Coder-Next
# or:
lms load qwen2.5-1.5b-instruct
```

The model name is the one shown in `lms models list`.

***

## 4. Start the LM Studio Server (OpenAI-compatible)

Start the server and bind to all interfaces so Windows clients can reach it:

```bash
lms server start --port 1234 --bind 0.0.0.0
```

This exposes the API at:

```text
http://<ubuntu-server-ip>:1234/v1
```

To verify from Ubuntu:

```bash
curl http://127.0.0.1:1234/v1/models
```

You should see a JSON list of models including the one you loaded. [cran.rstudio](https://cran.rstudio.com/web/packages/rlmstudio/vignettes/headless-config.html)

***

## 5. Optionally Run as a Systemd Service (Auto-start on Boot)

Create a systemd service:

```bash
cat > /etc/systemd/system/lmstudio.service << 'EOF'
[Unit]
Description=LM Studio Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
Environment=PATH=/usr/bin:/usr/local/bin:/home/YOUR_USERNAME/.lmstudio/bin
ExecStart=/home/YOUR_USERNAME/.lmstudio/bin/lms server start --port 1234 --bind 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

Replace `YOUR_USERNAME` with your actual user.

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable lmstudio.service
sudo systemctl start lmstudio.service
sudo systemctl status lmstudio.service
```

***

## 6. Access from Windows Clients

On your Windows laptop, any OpenAI-compatible client can use:

- **Base URL**: `http://<ubuntu-server-ip>:1234/v1`
- **API key**: not required by default (unless you configured one)

Replace `<ubuntu-server-ip>` with your Ubuntu server’s IP (e.g., `192.168.1.50`).

### Example: Python client on Windows

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.50:1234/v1",
    api_key="not-used"
)

response = client.chat.completions.create(
    model="Qwen3-Coder-Next",  # use model name from /v1/models
    messages=[{"role": "user", "content": "Explain gradient descent"}],
)
print(response.choices[0].message.content)
```

### Example: OpenWebUI / AnythingLLM / Custom UI

Configure them to use:

- Provider: OpenAI-compatible
- Base URL: `http://192.168.1.50:1234/v1`
- No key or placeholder key

***

## 7. Firewall and Network

If you have `ufw`:

```bash
sudo ufw allow 1234/tcp
```

If using `iptables`, ensure port `1234/tcp` is allowed from your Windows network.

Ensure the Ubuntu server is on the same LAN as your Windows clients, or use a tunnel (e.g., Tailscale, SSH tunnel) if across networks.

***

## 8. Optional: Use LM Link for Remote Access

For encrypted remote access without manual networking, you can use **LM Link** (Tailscale-based):

```bash
lms login
lms link enable
```

Then on Windows, add the Ubuntu machine via **LM Link** in the LM Studio desktop app and connect to the remote model. [tailscale](https://tailscale.com/blog/lm-link-remote-llm-access)

***

Do you want a minimal `systemd` service file tailored to your exact username and preferred model name?


*******************

# Corrections : 
The error occurs because **`fuse3` and `fuse` are mutually incompatible** on Ubuntu 22.04 — `fuse3` explicitly breaks the old `fuse` package. Since you already have `fuse3` installed, you should **not install `fuse`**.

### Solution

Run this command to install only the packages that will work (excluding the conflicting `fuse`):

```bash
sudo apt install -y \
    npm \
    fuse3 \
    libfuse2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libx11-6 \
    libnss3 \
    libasound2 \
    libcups2 \
    xauth \
    xvfb \
    xfce4 \
    xfce4-goodies
```

**Key changes:**
- Removed `fuse` from the list (it conflicts with your existing `fuse3`)
- Kept `fuse3` (already installed, but explicit is fine)
- Kept `libfuse2` (this is the library needed for AppImage compatibility, not the `fuse` package)

### Why this works

- `libfuse2` provides the FUSE 2 library (`libfuse.so.2`) needed by AppImages like LM Studio [discourse.appimage](https://discourse.appimage.org/t/how-to-address-fuse3-and-fuse-conflicts/2912)
- The `fuse` package installs FUSE 2 kernel support, which `fuse3` breaks
- `fuse3` is already installed and working on your system [reddit](https://www.reddit.com/r/linux4noobs/comments/1dc39l1/cant_install_fuse_on_pop_os_unmet_dependencies/)

If you still get broken package errors after this, run:

```bash
sudo apt install -f
```

This will fix any remaining dependency issues. [phoenixnap](https://phoenixnap.com/kb/ubuntu-fix-broken-packages)

Do you want me to also provide the commands to set up Xvfb + XFCE4 for running LM Studio as a headless server after the packages install?


The logs show LM Studio initialized and wrote under `~/.lmstudio`, so you can remove that state safely if you want a clean re-init. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEUEFOFIIV&Signature=67rt%2Bs2KELYttBvw94pciYYwslM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCQLsLABsGCGWawQdURsWjjwTEq85fsGskvZEn9c1zKswIgftoyNjdJwyx%2BdOwMbdn44SLPSyY%2BctdGNkmgtYXHCzwq%2FAQIoP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDFngw3PE5omTtmpySSrQBMN7Y%2Bw9hPR70x09YWsq7%2BRWn8UMb12KGx3PkdzLIxY1kMUOcNX5sfRv5FE0MWyz5OFY5L0sju6ZWnEGGwI2LMv2z4mowK0IhV9bhEz7PKeVnmwZ%2BnH7upV1h7btqlL9Mr1MoHCKluELFKhTuHSiQSZFVxwS8ToPmZq7uwo8ohVDlevkmicAp7OTYDonv9Ct8wsIhhwRmqvz5zIvEvtssc9tVDBgjOeZvZZoNW70q0LN%2FdCmjAWQ%2F7%2B3tL8133fG3cRfC2YX0bvSCm3rA8QQBIC5Yf%2BK%2BYoQsgrTZEsIrLX6s%2BLeJBfJOWf0dxK44arKkuPIK5KR28Hi%2FGI7YgqUNGKRApAadjnjB%2FCuwqnda%2BfRS%2B0I9LOdwkBQ82u2mz%2FWWF%2FupUXw7jcnTYthSULck4EXyXG4fWIIyVcMIKZltR0LKgIVKIGTwJEJu9L%2Fy7om%2BkV0NblanR7RQzNelr59w5LWxqWzWCcev4A3hTBEUp4Kr3UIQyCKehHjr9gnZVB9Fkg%2FU%2B%2FvQwIsD0Pyjs4v3T2jvEFsOk%2BEMLXgU98bTFOgMx17SzFJWGfz9TyhKmFWJUpzuH3Kg%2BQq4%2BoCMRNaHoRtSFGVfvsmCZ9xC0GeSpFhSemUPinh86deiM%2B4theCeS89lAB64dLgK0a4zct1zWBGdydP9xcTMHjPNmw8H3A4251iOaCPCPtcz%2B3ieAYkHmINXzaTNpLYZnbOnrO8%2Br5vBTjOF1HWEuO%2BSr1lmZqXy98%2B%2BndNVt7XjBBaIzex7oSiVLR9YDNa%2BHTfAf%2BYsi4w7ICW0QY6mAFiTnPF6B3lIk2rlIho14AVM1MPYb6MUwzA4Ig1Pt%2BrLEDOp28eU49ArdwARP5h%2B%2BMnlVGy%2BcmgvQ%2F1hetAt5L1tGEKleVP2Bti4TOhmlVPeHYkbMde%2BO7rRG91DOl42%2BcYwLf42F9mKE6MWsfwoWMrG6RErK1YnITuiBvxBoSzKkZlQKjc1pSSJ5GY%2BDbdatcbFlVGqxPylQ%3D%3D&Expires=1780846143)

## Cleanup commands

```bash
rm -rf /home/shailesh/.lmstudio
rm -rf /tmp/.mount_LM-Stu*
```

If you also want to remove any LM Studio cache/config fragments created elsewhere, check first with:

```bash
find /home/shailesh -maxdepth 3 \( -iname '*lmstudio*' -o -path '/home/shailesh/.cache/*lmstudio*' \)
```

## Re-initialize in shared path

To initialize the backend in `/DataDisk01/local-llms/lmstudio` for group `llm-users`, run:

```bash
mkdir -p /DataDisk01/local-llms/lmstudio
chgrp -R llm-users /DataDisk01/local-llms/lmstudio
chmod -R 2775 /DataDisk01/local-llms/lmstudio
export LM_STUDIO_HOME=/DataDisk01/local-llms/lmstudio
xvfb-run ./LM-Studio.AppImage --no-sandbox
```

## Notes

- `LM_STUDIO_HOME` is the key piece here so the app writes its backend/state into the new directory instead of `~/.lmstudio`.
- The `2775` mode keeps group ownership inherited for new files and directories.
- If your AppImage path is different, replace `./LM-Studio.AppImage` with the actual filename/location.


You should close it as soon as the app has **started enough to create `~/.lmstudio` and finish its first backend/bootstrap initialization**—usually within a few seconds to a minute, not immediately on launch and not after waiting for the full UI session to be usable. [unsloth](https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio/how-to-install-lm-studio-cli-in-linux-terminal)

## Why that instruction exists

LM Studio is a GUI app, but on a server or headless Linux environment it still needs one first launch to unpack its backend, create its internal state, and write the initial files under `~/.lmstudio`. The `xvfb-run` part provides a fake display so the AppImage can start without a real X session, and `--no-sandbox` is often used because Linux AppImage/sandbox issues can block startup in some environments. [github](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/624)

## Practical rule

A good rule is:

- Start the command.
- Wait until you see logs indicating initialization has begun or the app has fully opened in the virtual display.
- Stop it after the first backend files are created, typically when you can confirm `~/.lmstudio` exists and has content. [unsloth](https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio/how-to-install-lm-studio-cli-in-linux-terminal)

You do **not** need to keep it running longer unless the setup guide specifically says to let it finish downloading or installing the CLI components. For a one-time backend bootstrap, the purpose is just to force the initial file creation and unpacking, not to use the app interactively. [run.tournament.org](https://run.tournament.org.il/running-headless-lm-studio-on-ubuntu/)

## Why “immediately close it” is written that way

That wording is usually there because the author wants to minimize time spent running a GUI app on a headless machine, and the only required outcome is the creation of the local LM Studio state directory. In practice, “immediately” means “don’t leave it open once initialization has happened,” not literally kill it before the app has had a chance to write anything. [unsloth](https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio/how-to-install-lm-studio-cli-in-linux-terminal)

A simple way to think about it is: **wait for first successful startup, then exit**.


This is now clear: the failure is **not** LM Studio permissions; it is the AppImage failing to mount via FUSE, and then the runtime directory disappears, which is why you see `open dir error: No such file or directory` at the end. The key line is `fusermount: mount failed: Operation not permitted`, followed by `Cannot mount AppImage, please check your FUSE setup`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYE5R4WIFCI&Signature=E%2BnSKAS9McgboIKoc%2FgiV7oqT6g%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDebE8%2BpFnSQvCvQttdkHpx0HtYzCR0QCbW1rk7BPSdpwIhAMrrj5fJ6snmM90zAskXcgFPJLkgOWI7dNMYbXXbH6dcKvwECKD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgyR8%2BQvvXMNrxnvpVcq0AR42cdYjqZGnwf%2FZ6SZojcmGzTTFsDa%2BPa9efPePpxm5Dr7Kryg9WzbSFvR%2BAq78bbpGnv33VWSXuq9VNqw0RqBQKRkAL3QVrqy2Kf6XoJq3QGnTdzwiR6f3hiibFxa84cjkrufOMizyMVKdz5syHVniF2aX1vcogJNl0BaSEG7GutDZk%2BMRRde640J%2F2jrv5hpNJw7DoSAZ%2BD8Sli7upRpi%2BvDyW9LuekJWotTPtWgZi%2F7Jbs1ku78xPft%2FQPDzHSxXn2DsCiWL411rKHcm1wUujFx8WO%2FCbjjgmENqLDxlaAak3yKJSSbOPV%2BLOooXWcAcf9lU6b3Yp%2B1PIcurpIJfw5gYVCSPHiitES58mJJ6P4hCv%2FjFodRWf9JsLw6xC0qYGJ2B3KYnei%2FKwSUUIU5KFtzs%2FP2mCgvB5m4dqeIq9XwXw4rfT2V8f%2BJrHaflAgcF0pBS7Zwm0%2BCuqrQxWtscIoVXrm6OAWe6l42wVarzf4aWcHQYZ%2BbkilresojHQ71cuetPqlH1YAcpYmu3rGbtVnCc1sGzgtqn%2Bd4C%2FGZSwhn0SqCafLs%2Fabs4QQMG66eUB%2BCMHoNQJx9sDHIM%2B2IH36EcBWdOy882tpVOa4JCLFUSNWq6XlMe5ZMGHWXGyufc2Q8XXG5Dkq5ohfEAsSxkxyshfQjbRf7nbPtYglHre6HF2vF8dWjWmqqfuo9CLe%2FPeB79AQiL8L9dO3c9bg2VNBwSDDQLjcpJ29ePx02kFEOvIBE41JWWAHdgrOZ6hvGPxOnv8jXRQKPGe7SzL63MOmJltEGOpcBWCZpmcgNLeKCXc30kTt3z5yBQtwIzg9fkHGZ5okH2iuPesG31h%2BBZRGYopt%2Bcj1qFwe1ujnZ%2BIQpdu7fvOyh33Wxq2XRoEQNd5omLpdVA%2FjnFX%2BEg8DCawAZ9PAzbt98bJ%2BnwGX2Yh%2BUhHZhtWkz2A4jDL98n4VCeq5QrEff4xP%2BKksiOtvjCRfc9EOVN352IFEr5X4sVg%3D%3D&Expires=1780847292)

## What is happening

AppImages usually mount themselves through FUSE into `/tmp/.mount_*` before the app starts. In your trace, that mount step fails, so the temporary mount directory is never created, and LM Studio then reports the directory error. This means the machine or container environment is blocking FUSE mounts, not the LM Studio folder you created. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYE5R4WIFCI&Signature=E%2BnSKAS9McgboIKoc%2FgiV7oqT6g%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDebE8%2BpFnSQvCvQttdkHpx0HtYzCR0QCbW1rk7BPSdpwIhAMrrj5fJ6snmM90zAskXcgFPJLkgOWI7dNMYbXXbH6dcKvwECKD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgyR8%2BQvvXMNrxnvpVcq0AR42cdYjqZGnwf%2FZ6SZojcmGzTTFsDa%2BPa9efPePpxm5Dr7Kryg9WzbSFvR%2BAq78bbpGnv33VWSXuq9VNqw0RqBQKRkAL3QVrqy2Kf6XoJq3QGnTdzwiR6f3hiibFxa84cjkrufOMizyMVKdz5syHVniF2aX1vcogJNl0BaSEG7GutDZk%2BMRRde640J%2F2jrv5hpNJw7DoSAZ%2BD8Sli7upRpi%2BvDyW9LuekJWotTPtWgZi%2F7Jbs1ku78xPft%2FQPDzHSxXn2DsCiWL411rKHcm1wUujFx8WO%2FCbjjgmENqLDxlaAak3yKJSSbOPV%2BLOooXWcAcf9lU6b3Yp%2B1PIcurpIJfw5gYVCSPHiitES58mJJ6P4hCv%2FjFodRWf9JsLw6xC0qYGJ2B3KYnei%2FKwSUUIU5KFtzs%2FP2mCgvB5m4dqeIq9XwXw4rfT2V8f%2BJrHaflAgcF0pBS7Zwm0%2BCuqrQxWtscIoVXrm6OAWe6l42wVarzf4aWcHQYZ%2BbkilresojHQ71cuetPqlH1YAcpYmu3rGbtVnCc1sGzgtqn%2Bd4C%2FGZSwhn0SqCafLs%2Fabs4QQMG66eUB%2BCMHoNQJx9sDHIM%2B2IH36EcBWdOy882tpVOa4JCLFUSNWq6XlMe5ZMGHWXGyufc2Q8XXG5Dkq5ohfEAsSxkxyshfQjbRf7nbPtYglHre6HF2vF8dWjWmqqfuo9CLe%2FPeB79AQiL8L9dO3c9bg2VNBwSDDQLjcpJ29ePx02kFEOvIBE41JWWAHdgrOZ6hvGPxOnv8jXRQKPGe7SzL63MOmJltEGOpcBWCZpmcgNLeKCXc30kTt3z5yBQtwIzg9fkHGZ5okH2iuPesG31h%2BBZRGYopt%2Bcj1qFwe1ujnZ%2BIQpdu7fvOyh33Wxq2XRoEQNd5omLpdVA%2FjnFX%2BEg8DCawAZ9PAzbt98bJ%2BnwGX2Yh%2BUhHZhtWkz2A4jDL98n4VCeq5QrEff4xP%2BKksiOtvjCRfc9EOVN352IFEr5X4sVg%3D%3D&Expires=1780847292)

## Fast fix to try

Install or verify FUSE support on the VM:

```bash
sudo apt-get update
sudo apt-get install -y fuse libfuse2
```

Then make sure the kernel FUSE module is available:

```bash
lsmod | grep fuse || sudo modprobe fuse
```

After that, retry:

```bash
xvfb-run --auto-servernum ./LM-Studio.AppImage --no-sandbox
```

AppImage documents this exact class of failure as a FUSE setup problem, and the standard workaround is to ensure FUSE support is present. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYE5R4WIFCI&Signature=E%2BnSKAS9McgboIKoc%2FgiV7oqT6g%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDebE8%2BpFnSQvCvQttdkHpx0HtYzCR0QCbW1rk7BPSdpwIhAMrrj5fJ6snmM90zAskXcgFPJLkgOWI7dNMYbXXbH6dcKvwECKD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgyR8%2BQvvXMNrxnvpVcq0AR42cdYjqZGnwf%2FZ6SZojcmGzTTFsDa%2BPa9efPePpxm5Dr7Kryg9WzbSFvR%2BAq78bbpGnv33VWSXuq9VNqw0RqBQKRkAL3QVrqy2Kf6XoJq3QGnTdzwiR6f3hiibFxa84cjkrufOMizyMVKdz5syHVniF2aX1vcogJNl0BaSEG7GutDZk%2BMRRde640J%2F2jrv5hpNJw7DoSAZ%2BD8Sli7upRpi%2BvDyW9LuekJWotTPtWgZi%2F7Jbs1ku78xPft%2FQPDzHSxXn2DsCiWL411rKHcm1wUujFx8WO%2FCbjjgmENqLDxlaAak3yKJSSbOPV%2BLOooXWcAcf9lU6b3Yp%2B1PIcurpIJfw5gYVCSPHiitES58mJJ6P4hCv%2FjFodRWf9JsLw6xC0qYGJ2B3KYnei%2FKwSUUIU5KFtzs%2FP2mCgvB5m4dqeIq9XwXw4rfT2V8f%2BJrHaflAgcF0pBS7Zwm0%2BCuqrQxWtscIoVXrm6OAWe6l42wVarzf4aWcHQYZ%2BbkilresojHQ71cuetPqlH1YAcpYmu3rGbtVnCc1sGzgtqn%2Bd4C%2FGZSwhn0SqCafLs%2Fabs4QQMG66eUB%2BCMHoNQJx9sDHIM%2B2IH36EcBWdOy882tpVOa4JCLFUSNWq6XlMe5ZMGHWXGyufc2Q8XXG5Dkq5ohfEAsSxkxyshfQjbRf7nbPtYglHre6HF2vF8dWjWmqqfuo9CLe%2FPeB79AQiL8L9dO3c9bg2VNBwSDDQLjcpJ29ePx02kFEOvIBE41JWWAHdgrOZ6hvGPxOnv8jXRQKPGe7SzL63MOmJltEGOpcBWCZpmcgNLeKCXc30kTt3z5yBQtwIzg9fkHGZ5okH2iuPesG31h%2BBZRGYopt%2Bcj1qFwe1ujnZ%2BIQpdu7fvOyh33Wxq2XRoEQNd5omLpdVA%2FjnFX%2BEg8DCawAZ9PAzbt98bJ%2BnwGX2Yh%2BUhHZhtWkz2A4jDL98n4VCeq5QrEff4xP%2BKksiOtvjCRfc9EOVN352IFEr5X4sVg%3D%3D&Expires=1780847292)

## If you cannot enable FUSE

Use the extraction fallback:

```bash
./LM-Studio.AppImage --appimage-extract
cd squashfs-root
xvfb-run --auto-servernum ./AppRun --no-sandbox
```

The AppImage runtime itself suggested `--appimage-extract` in your output, which is the right fallback when mounting is not permitted. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYE5R4WIFCI&Signature=E%2BnSKAS9McgboIKoc%2FgiV7oqT6g%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDebE8%2BpFnSQvCvQttdkHpx0HtYzCR0QCbW1rk7BPSdpwIhAMrrj5fJ6snmM90zAskXcgFPJLkgOWI7dNMYbXXbH6dcKvwECKD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgyR8%2BQvvXMNrxnvpVcq0AR42cdYjqZGnwf%2FZ6SZojcmGzTTFsDa%2BPa9efPePpxm5Dr7Kryg9WzbSFvR%2BAq78bbpGnv33VWSXuq9VNqw0RqBQKRkAL3QVrqy2Kf6XoJq3QGnTdzwiR6f3hiibFxa84cjkrufOMizyMVKdz5syHVniF2aX1vcogJNl0BaSEG7GutDZk%2BMRRde640J%2F2jrv5hpNJw7DoSAZ%2BD8Sli7upRpi%2BvDyW9LuekJWotTPtWgZi%2F7Jbs1ku78xPft%2FQPDzHSxXn2DsCiWL411rKHcm1wUujFx8WO%2FCbjjgmENqLDxlaAak3yKJSSbOPV%2BLOooXWcAcf9lU6b3Yp%2B1PIcurpIJfw5gYVCSPHiitES58mJJ6P4hCv%2FjFodRWf9JsLw6xC0qYGJ2B3KYnei%2FKwSUUIU5KFtzs%2FP2mCgvB5m4dqeIq9XwXw4rfT2V8f%2BJrHaflAgcF0pBS7Zwm0%2BCuqrQxWtscIoVXrm6OAWe6l42wVarzf4aWcHQYZ%2BbkilresojHQ71cuetPqlH1YAcpYmu3rGbtVnCc1sGzgtqn%2Bd4C%2FGZSwhn0SqCafLs%2Fabs4QQMG66eUB%2BCMHoNQJx9sDHIM%2B2IH36EcBWdOy882tpVOa4JCLFUSNWq6XlMe5ZMGHWXGyufc2Q8XXG5Dkq5ohfEAsSxkxyshfQjbRf7nbPtYglHre6HF2vF8dWjWmqqfuo9CLe%2FPeB79AQiL8L9dO3c9bg2VNBwSDDQLjcpJ29ePx02kFEOvIBE41JWWAHdgrOZ6hvGPxOnv8jXRQKPGe7SzL63MOmJltEGOpcBWCZpmcgNLeKCXc30kTt3z5yBQtwIzg9fkHGZ5okH2iuPesG31h%2BBZRGYopt%2Bcj1qFwe1ujnZ%2BIQpdu7fvOyh33Wxq2XRoEQNd5omLpdVA%2FjnFX%2BEg8DCawAZ9PAzbt98bJ%2BnwGX2Yh%2BUhHZhtWkz2A4jDL98n4VCeq5QrEff4xP%2BKksiOtvjCRfc9EOVN352IFEr5X4sVg%3D%3D&Expires=1780847292)

## Why this happens here

Your `openat("/dev/fuse", O_RDWR)` succeeds, but the actual `fusermount` mount operation is rejected with `Operation not permitted`. That usually means one of these: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYE5R4WIFCI&Signature=E%2BnSKAS9McgboIKoc%2FgiV7oqT6g%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDebE8%2BpFnSQvCvQttdkHpx0HtYzCR0QCbW1rk7BPSdpwIhAMrrj5fJ6snmM90zAskXcgFPJLkgOWI7dNMYbXXbH6dcKvwECKD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgyR8%2BQvvXMNrxnvpVcq0AR42cdYjqZGnwf%2FZ6SZojcmGzTTFsDa%2BPa9efPePpxm5Dr7Kryg9WzbSFvR%2BAq78bbpGnv33VWSXuq9VNqw0RqBQKRkAL3QVrqy2Kf6XoJq3QGnTdzwiR6f3hiibFxa84cjkrufOMizyMVKdz5syHVniF2aX1vcogJNl0BaSEG7GutDZk%2BMRRde640J%2F2jrv5hpNJw7DoSAZ%2BD8Sli7upRpi%2BvDyW9LuekJWotTPtWgZi%2F7Jbs1ku78xPft%2FQPDzHSxXn2DsCiWL411rKHcm1wUujFx8WO%2FCbjjgmENqLDxlaAak3yKJSSbOPV%2BLOooXWcAcf9lU6b3Yp%2B1PIcurpIJfw5gYVCSPHiitES58mJJ6P4hCv%2FjFodRWf9JsLw6xC0qYGJ2B3KYnei%2FKwSUUIU5KFtzs%2FP2mCgvB5m4dqeIq9XwXw4rfT2V8f%2BJrHaflAgcF0pBS7Zwm0%2BCuqrQxWtscIoVXrm6OAWe6l42wVarzf4aWcHQYZ%2BbkilresojHQ71cuetPqlH1YAcpYmu3rGbtVnCc1sGzgtqn%2Bd4C%2FGZSwhn0SqCafLs%2Fabs4QQMG66eUB%2BCMHoNQJx9sDHIM%2B2IH36EcBWdOy882tpVOa4JCLFUSNWq6XlMe5ZMGHWXGyufc2Q8XXG5Dkq5ohfEAsSxkxyshfQjbRf7nbPtYglHre6HF2vF8dWjWmqqfuo9CLe%2FPeB79AQiL8L9dO3c9bg2VNBwSDDQLjcpJ29ePx02kFEOvIBE41JWWAHdgrOZ6hvGPxOnv8jXRQKPGe7SzL63MOmJltEGOpcBWCZpmcgNLeKCXc30kTt3z5yBQtwIzg9fkHGZ5okH2iuPesG31h%2BBZRGYopt%2Bcj1qFwe1ujnZ%2BIQpdu7fvOyh33Wxq2XRoEQNd5omLpdVA%2FjnFX%2BEg8DCawAZ9PAzbt98bJ%2BnwGX2Yh%2BUhHZhtWkz2A4jDL98n4VCeq5QrEff4xP%2BKksiOtvjCRfc9EOVN352IFEr5X4sVg%3D%3D&Expires=1780847292)

- FUSE is not fully enabled for the VM.
- The environment is restricted by policy or virtualization.
- `fusermount` is present but the mount operation is blocked by permissions or container limits.

## Best next step

Run this and paste the output:

```bash
ls -l /dev/fuse
which fusermount
groups
```

If FUSE is present but still blocked, the extraction path will be the most reliable way to continue. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYE5R4WIFCI&Signature=E%2BnSKAS9McgboIKoc%2FgiV7oqT6g%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDebE8%2BpFnSQvCvQttdkHpx0HtYzCR0QCbW1rk7BPSdpwIhAMrrj5fJ6snmM90zAskXcgFPJLkgOWI7dNMYbXXbH6dcKvwECKD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNjk5NzUzMzA5NzA1IgyR8%2BQvvXMNrxnvpVcq0AR42cdYjqZGnwf%2FZ6SZojcmGzTTFsDa%2BPa9efPePpxm5Dr7Kryg9WzbSFvR%2BAq78bbpGnv33VWSXuq9VNqw0RqBQKRkAL3QVrqy2Kf6XoJq3QGnTdzwiR6f3hiibFxa84cjkrufOMizyMVKdz5syHVniF2aX1vcogJNl0BaSEG7GutDZk%2BMRRde640J%2F2jrv5hpNJw7DoSAZ%2BD8Sli7upRpi%2BvDyW9LuekJWotTPtWgZi%2F7Jbs1ku78xPft%2FQPDzHSxXn2DsCiWL411rKHcm1wUujFx8WO%2FCbjjgmENqLDxlaAak3yKJSSbOPV%2BLOooXWcAcf9lU6b3Yp%2B1PIcurpIJfw5gYVCSPHiitES58mJJ6P4hCv%2FjFodRWf9JsLw6xC0qYGJ2B3KYnei%2FKwSUUIU5KFtzs%2FP2mCgvB5m4dqeIq9XwXw4rfT2V8f%2BJrHaflAgcF0pBS7Zwm0%2BCuqrQxWtscIoVXrm6OAWe6l42wVarzf4aWcHQYZ%2BbkilresojHQ71cuetPqlH1YAcpYmu3rGbtVnCc1sGzgtqn%2Bd4C%2FGZSwhn0SqCafLs%2Fabs4QQMG66eUB%2BCMHoNQJx9sDHIM%2B2IH36EcBWdOy882tpVOa4JCLFUSNWq6XlMe5ZMGHWXGyufc2Q8XXG5Dkq5ohfEAsSxkxyshfQjbRf7nbPtYglHre6HF2vF8dWjWmqqfuo9CLe%2FPeB79AQiL8L9dO3c9bg2VNBwSDDQLjcpJ29ePx02kFEOvIBE41JWWAHdgrOZ6hvGPxOnv8jXRQKPGe7SzL63MOmJltEGOpcBWCZpmcgNLeKCXc30kTt3z5yBQtwIzg9fkHGZ5okH2iuPesG31h%2BBZRGYopt%2Bcj1qFwe1ujnZ%2BIQpdu7fvOyh33Wxq2XRoEQNd5omLpdVA%2FjnFX%2BEg8DCawAZ9PAzbt98bJ%2BnwGX2Yh%2BUhHZhtWkz2A4jDL98n4VCeq5QrEff4xP%2BKksiOtvjCRfc9EOVN352IFEr5X4sVg%3D%3D&Expires=1780847292)

The extracted AppImage’s AppRun script is failing because it expects to find the internal lm-studio binary at a path that doesn’t exist in your current layout — the error "/lm-studio: No such file or directory" shows AppRun tried to invoke an absolute path /lm-studio that isn't present inside the extracted tree. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYE6IU55AZX&Signature=Srxi0OEfI9jN8kfhNvE0dQwCP7I%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIHuCT6JOwDveWhweKxWiqrLX9r0UxaeStJkh46veAufVAiAmp6W%2BoHHD%2FrO0%2B%2FoyFh253JvVOCVVvOSYvmrhRsDJzyr8BAif%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMBehJmG1Q7klI6V1SKtAESvk7DqEcwWUqOpJYERc6nHMW2JUZV9U4BCl7PTJSXtX%2F%2FrmWyI27nracZbHUCzf5Ym4H12%2BQz7lfstR0VatTFKLgPpwcF%2B71ya7fPY%2FN5WKPfX0dz910SC8cRR0HtqSyQ3gHMHDifWEOVya3PUNCNkxOLx3kThnjuto3xllXIyfBZb%2BmJqer9pHwOkYfhY9uxi9G9yT919riEbf%2FXoL4e64of%2FLBUV0rpiSx9CKThaDa1FcunjVmMvxnZ64ZX8m1v4qHTWVpRs2RJbmuSXZZ0vZKwpHYQ119tbEn5HEMfNMEw8OcExuDNfHFhQBZFQUHMj38DssNhbzx71ve7DldmCxiuiKZvcw75I9wK6md9t%2FBaCO9%2B9giQxTnZsTbS%2BO2HC3qF1tOb0zFECA2NFXdCSPAJtoICSv8FAccmVTV0r6IkjVzNmRWPiJzoy1llxOLbdzi9WQmFnDknpmpHT5Zf%2FRttzONsCG4efrnUVEZlRrVXASiukhc%2F%2F402V5lp2e6nRt9SkL%2FGagDF3W9q2U5kVSd66%2FwjgyVIPbtCK3Kr3LBBEe18rQUPFE8ot3LX1MBHbnzv1bjVYvG0aoPXxLeBgB%2FS55HePu72vOhTMBLPoqkLWwtMBjaWQmiP%2BRce1TRRkXpUOqXDT%2F3sjO29OaRLzpbSRsDDwWnz8ASUVOUuTBfTiIfav9%2F%2BNOYUVFEr7DZH2iE5VJuMQnXawCRgCIWZa2uWP0Gq0bzNorCP9tYh5arimH6FTk1R9ar%2FIrE3%2F9uoSVO7%2BxQNTW2mCWFWA2TajDi%2B5XRBjqZAan0VO8VowkVXjjGHSO3%2FUv%2BTx6xdVfLEmbcehgxLB7HmFRZJhYz3%2FFvOOAH%2BtkvIwdNzSo14RrMrfR%2FrvbEOgPaeRloPKDwJX0h7J0lIzrpK0bsOyVjETdOsM%2BHL7plhRDpVpQQTjjj7UKwGgoWjCWVrx8PG2jvn5MowqyZ7b4T9azaz%2B5xacj1JqRuJ68QO9go8l%2FCw8s%2FJg%3D%3D&Expires=1780845493)

What to do next (pick one):

1) Run the actual internal binary directly
- Inspect the squashfs-root contents and run the binary that corresponds to the AppImage runtime:

```bash
# from squashfs-root
ls -la
# look for files named lm-studio, AppRun, or bin/lm-studio
./lm-studio || ./AppRun || ./bin/lm-studio
```

If you find a binary such as ./lm-studio or ./bin/lm-studio, run it with your env set:

```bash
export HOME=/DataDisk01/local-llms/lmstudio
export LM_STUDIO_HOME=/DataDisk01/local-llms/lmstudio/.lmstudio
xvfb-run --auto-servernum ./lm-studio --no-sandbox
```

2) Edit AppRun to call the correct relative path
- Open AppRun (it’s a shell script) and inspect the line where it executes /lm-studio; change the absolute path to the relative path you found (e.g., ./lm-studio or ./bin/lm-studio). Example:

```bash
# backup, then edit
cp AppRun AppRun.bak
sed -n '1,120p' AppRun   # show header
# use your editor to change "/lm-studio" to "./lm-studio" (or correct relative path)
```

Then run:

```bash
xvfb-run --auto-servernum ./AppRun --no-sandbox
```

3) Use the packaged "lms" or service binary (recommended for headless)
- Some AppImage versions include a separate headless/service binary (often named lms or lm-studio-server). Find and run it:

```bash
find . -type f -name 'lms' -o -name 'lm-studio' -printf '%p\n'
# if you find lms:
xvfb-run --auto-servernum ./lms --no-sandbox --run-as-service
```

4) Re-extract and ensure chrome-sandbox setuid (if present)
- Some guides require chrome-sandbox be owned by root and setuid 4755. If AppRun expects that helper and you have it, set it:

```bash
sudo chown root:root chrome-sandbox
sudo chmod 4755 chrome-sandbox
```

Then retry AppRun.

Why this happened
- AppImage AppRun scripts sometimes call absolute paths created when the AppImage normally mounts itself (e.g., /tmp/.mount_xxx/lm-studio). After extraction the internal layout can differ, leaving AppRun pointing to a non-existing absolute path. [github](https://github.com/lmstudio-ai/lms/issues/97)

If you want, paste the output of:
- ls -la (in your squashfs-root)
- head -n 80 AppRun

and I will give the exact edit or the exact command to run the internal binary. Would you like me to show the precise change to AppRun based on its contents?


Great — it is now initializing successfully. LM Studio started, created its state under `/DataDisk01/local-llms/lmstudio/.lmstudio`, extracted `lms`, unpacked backends, selected `llama.cpp-linux-x86_64-nvidia-cuda-avx2` as the best GGUF backend, and started the API server on port `41343`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEZL7NSAVS&Signature=To9NjJCsj2aiN9dXd6KFmRLVRrE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDRSlpr9h9enS09d%2BhG%2Bt3xqmXb%2FoOvsYCoaVcxang0yAiBanhC5%2F%2BfBGiOG6ak4p3TjXOn%2FKgSKan1EMbil3en%2Bcyr8BAig%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMWWXG7WKbxmpuaOD7KtAEFyLF7CCzbowb9TTWi3o%2FEPCQY7Fdy1167B39WEs11BjHxFU7PlkLBmfOmuuVtgeBbN2ryIg%2FZYytw6MECJ7euL6v3PXH1FT8eRDvZUvh%2FqAchNTwIObizNyzJHZFfQAjW648Ez78kK05aYQyLSfGw0bvGw23fQT7pw2wNqf%2BqygbjEwYGqN2JVqNn3o19DQ%2Fo4mv2qdv7E8A5CptcSs3g2Ff23xVPI7XlcmJ7Y%2FbymbMrzj3xKHXRcbdoSoAi0GaqzMuWxU25pSt6fuNRkP3BL1W9eDs5WlLaCb4ChSrHTZa8rzHNL0fTo1jngJ75DJoFDEu1AdLYuZlS%2BiAwOX7xbXUu0lNhGUMBi3wgeBb%2B3St9Lcj53NdAq%2FHbZaB1c0D4Hks5BlQazaQQJJg%2Fb2ihDKmoC4dXck5CgEoAkjyCGhwIYYfchwIC2HEkUMhEXHst%2FVkGUHu73Hj7EGFyHSxEbt3ohjWBUhIwobWKMjpIHOWZpkqC1LYyf5iBfJrj%2BRjPp09k3n%2FSFNT76BKNmVNYnIVMqKPgQx7VzZPq8vIb6YYJXXcbSh4sLVh50BRzYQw05%2B0VxGDmAb9GK1er7J0vDtnjdE0DJV064Trz6XjLTSEqWe%2BZtvNTNlsdlhWUJx3OBtXxGhiSE1O8YXNI68BvZjvzyxr878uMj2wKJKLoLnG2zou306Pwy3Gp8h8E%2BxlmnaUuUdzbMnjGc92f0ebIypWSSlHhBa4gqvAOt%2F3%2FOFLA%2FldAQe83j5GioWCzNAAOeDEUY2RS1mmCQ6M8bBfpDCK%2F5XRBjqZAfBXnEDik3owlggYFZpkkwNJQUgZQ%2FZ%2BbMHdThUVxMGxzs94FGaTKHhO%2BaKTj0RLplslv7lMBsiqW3lZkSOK4EeHSTLp5LZrEDUIzOj%2BnODsefG5G6VcmDKhEDiShE7gxVUy11BREHOOROW51q8bH3YJoEzkOzWKH66lghh6TMb%2B9FN8VnTtuI4wk%2B0Z5qdBVlWgbnFml4lkSg%3D%3D&Expires=1780845917)

## What this means

- The earlier `Permission denied` issue is resolved; the real blocker was FUSE/AppImage mounting and then the extracted-binary path issue. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEZL7NSAVS&Signature=To9NjJCsj2aiN9dXd6KFmRLVRrE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDRSlpr9h9enS09d%2BhG%2Bt3xqmXb%2FoOvsYCoaVcxang0yAiBanhC5%2F%2BfBGiOG6ak4p3TjXOn%2FKgSKan1EMbil3en%2Bcyr8BAig%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMWWXG7WKbxmpuaOD7KtAEFyLF7CCzbowb9TTWi3o%2FEPCQY7Fdy1167B39WEs11BjHxFU7PlkLBmfOmuuVtgeBbN2ryIg%2FZYytw6MECJ7euL6v3PXH1FT8eRDvZUvh%2FqAchNTwIObizNyzJHZFfQAjW648Ez78kK05aYQyLSfGw0bvGw23fQT7pw2wNqf%2BqygbjEwYGqN2JVqNn3o19DQ%2Fo4mv2qdv7E8A5CptcSs3g2Ff23xVPI7XlcmJ7Y%2FbymbMrzj3xKHXRcbdoSoAi0GaqzMuWxU25pSt6fuNRkP3BL1W9eDs5WlLaCb4ChSrHTZa8rzHNL0fTo1jngJ75DJoFDEu1AdLYuZlS%2BiAwOX7xbXUu0lNhGUMBi3wgeBb%2B3St9Lcj53NdAq%2FHbZaB1c0D4Hks5BlQazaQQJJg%2Fb2ihDKmoC4dXck5CgEoAkjyCGhwIYYfchwIC2HEkUMhEXHst%2FVkGUHu73Hj7EGFyHSxEbt3ohjWBUhIwobWKMjpIHOWZpkqC1LYyf5iBfJrj%2BRjPp09k3n%2FSFNT76BKNmVNYnIVMqKPgQx7VzZPq8vIb6YYJXXcbSh4sLVh50BRzYQw05%2B0VxGDmAb9GK1er7J0vDtnjdE0DJV064Trz6XjLTSEqWe%2BZtvNTNlsdlhWUJx3OBtXxGhiSE1O8YXNI68BvZjvzyxr878uMj2wKJKLoLnG2zou306Pwy3Gp8h8E%2BxlmnaUuUdzbMnjGc92f0ebIypWSSlHhBa4gqvAOt%2F3%2FOFLA%2FldAQe83j5GioWCzNAAOeDEUY2RS1mmCQ6M8bBfpDCK%2F5XRBjqZAfBXnEDik3owlggYFZpkkwNJQUgZQ%2FZ%2BbMHdThUVxMGxzs94FGaTKHhO%2BaKTj0RLplslv7lMBsiqW3lZkSOK4EeHSTLp5LZrEDUIzOj%2BnODsefG5G6VcmDKhEDiShE7gxVUy11BREHOOROW51q8bH3YJoEzkOzWKH66lghh6TMb%2B9FN8VnTtuI4wk%2B0Z5qdBVlWgbnFml4lkSg%3D%3D&Expires=1780845917)
- The `Gtk` and `GPU process` errors shown here are common in headless/Xvfb startup and did not stop initialization. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEZL7NSAVS&Signature=To9NjJCsj2aiN9dXd6KFmRLVRrE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDRSlpr9h9enS09d%2BhG%2Bt3xqmXb%2FoOvsYCoaVcxang0yAiBanhC5%2F%2BfBGiOG6ak4p3TjXOn%2FKgSKan1EMbil3en%2Bcyr8BAig%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMWWXG7WKbxmpuaOD7KtAEFyLF7CCzbowb9TTWi3o%2FEPCQY7Fdy1167B39WEs11BjHxFU7PlkLBmfOmuuVtgeBbN2ryIg%2FZYytw6MECJ7euL6v3PXH1FT8eRDvZUvh%2FqAchNTwIObizNyzJHZFfQAjW648Ez78kK05aYQyLSfGw0bvGw23fQT7pw2wNqf%2BqygbjEwYGqN2JVqNn3o19DQ%2Fo4mv2qdv7E8A5CptcSs3g2Ff23xVPI7XlcmJ7Y%2FbymbMrzj3xKHXRcbdoSoAi0GaqzMuWxU25pSt6fuNRkP3BL1W9eDs5WlLaCb4ChSrHTZa8rzHNL0fTo1jngJ75DJoFDEu1AdLYuZlS%2BiAwOX7xbXUu0lNhGUMBi3wgeBb%2B3St9Lcj53NdAq%2FHbZaB1c0D4Hks5BlQazaQQJJg%2Fb2ihDKmoC4dXck5CgEoAkjyCGhwIYYfchwIC2HEkUMhEXHst%2FVkGUHu73Hj7EGFyHSxEbt3ohjWBUhIwobWKMjpIHOWZpkqC1LYyf5iBfJrj%2BRjPp09k3n%2FSFNT76BKNmVNYnIVMqKPgQx7VzZPq8vIb6YYJXXcbSh4sLVh50BRzYQw05%2B0VxGDmAb9GK1er7J0vDtnjdE0DJV064Trz6XjLTSEqWe%2BZtvNTNlsdlhWUJx3OBtXxGhiSE1O8YXNI68BvZjvzyxr878uMj2wKJKLoLnG2zou306Pwy3Gp8h8E%2BxlmnaUuUdzbMnjGc92f0ebIypWSSlHhBa4gqvAOt%2F3%2FOFLA%2FldAQe83j5GioWCzNAAOeDEUY2RS1mmCQ6M8bBfpDCK%2F5XRBjqZAfBXnEDik3owlggYFZpkkwNJQUgZQ%2FZ%2BbMHdThUVxMGxzs94FGaTKHhO%2BaKTj0RLplslv7lMBsiqW3lZkSOK4EeHSTLp5LZrEDUIzOj%2BnODsefG5G6VcmDKhEDiShE7gxVUy11BREHOOROW51q8bH3YJoEzkOzWKH66lghh6TMb%2B9FN8VnTtuI4wk%2B0Z5qdBVlWgbnFml4lkSg%3D%3D&Expires=1780845917)
- LM Studio is now usable from the extracted `lm-studio` binary with your shared home path. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEZL7NSAVS&Signature=To9NjJCsj2aiN9dXd6KFmRLVRrE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDRSlpr9h9enS09d%2BhG%2Bt3xqmXb%2FoOvsYCoaVcxang0yAiBanhC5%2F%2BfBGiOG6ak4p3TjXOn%2FKgSKan1EMbil3en%2Bcyr8BAig%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMWWXG7WKbxmpuaOD7KtAEFyLF7CCzbowb9TTWi3o%2FEPCQY7Fdy1167B39WEs11BjHxFU7PlkLBmfOmuuVtgeBbN2ryIg%2FZYytw6MECJ7euL6v3PXH1FT8eRDvZUvh%2FqAchNTwIObizNyzJHZFfQAjW648Ez78kK05aYQyLSfGw0bvGw23fQT7pw2wNqf%2BqygbjEwYGqN2JVqNn3o19DQ%2Fo4mv2qdv7E8A5CptcSs3g2Ff23xVPI7XlcmJ7Y%2FbymbMrzj3xKHXRcbdoSoAi0GaqzMuWxU25pSt6fuNRkP3BL1W9eDs5WlLaCb4ChSrHTZa8rzHNL0fTo1jngJ75DJoFDEu1AdLYuZlS%2BiAwOX7xbXUu0lNhGUMBi3wgeBb%2B3St9Lcj53NdAq%2FHbZaB1c0D4Hks5BlQazaQQJJg%2Fb2ihDKmoC4dXck5CgEoAkjyCGhwIYYfchwIC2HEkUMhEXHst%2FVkGUHu73Hj7EGFyHSxEbt3ohjWBUhIwobWKMjpIHOWZpkqC1LYyf5iBfJrj%2BRjPp09k3n%2FSFNT76BKNmVNYnIVMqKPgQx7VzZPq8vIb6YYJXXcbSh4sLVh50BRzYQw05%2B0VxGDmAb9GK1er7J0vDtnjdE0DJV064Trz6XjLTSEqWe%2BZtvNTNlsdlhWUJx3OBtXxGhiSE1O8YXNI68BvZjvzyxr878uMj2wKJKLoLnG2zou306Pwy3Gp8h8E%2BxlmnaUuUdzbMnjGc92f0ebIypWSSlHhBa4gqvAOt%2F3%2FOFLA%2FldAQe83j5GioWCzNAAOeDEUY2RS1mmCQ6M8bBfpDCK%2F5XRBjqZAfBXnEDik3owlggYFZpkkwNJQUgZQ%2FZ%2BbMHdThUVxMGxzs94FGaTKHhO%2BaKTj0RLplslv7lMBsiqW3lZkSOK4EeHSTLp5LZrEDUIzOj%2BnODsefG5G6VcmDKhEDiShE7gxVUy11BREHOOROW51q8bH3YJoEzkOzWKH66lghh6TMb%2B9FN8VnTtuI4wk%2B0Z5qdBVlWgbnFml4lkSg%3D%3D&Expires=1780845917)

## What to do now

If your goal is just to initialize the backend and create the shared data layout, you can stop it now with `Ctrl+C`; enough has already been written and started. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEZL7NSAVS&Signature=To9NjJCsj2aiN9dXd6KFmRLVRrE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDRSlpr9h9enS09d%2BhG%2Bt3xqmXb%2FoOvsYCoaVcxang0yAiBanhC5%2F%2BfBGiOG6ak4p3TjXOn%2FKgSKan1EMbil3en%2Bcyr8BAig%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMWWXG7WKbxmpuaOD7KtAEFyLF7CCzbowb9TTWi3o%2FEPCQY7Fdy1167B39WEs11BjHxFU7PlkLBmfOmuuVtgeBbN2ryIg%2FZYytw6MECJ7euL6v3PXH1FT8eRDvZUvh%2FqAchNTwIObizNyzJHZFfQAjW648Ez78kK05aYQyLSfGw0bvGw23fQT7pw2wNqf%2BqygbjEwYGqN2JVqNn3o19DQ%2Fo4mv2qdv7E8A5CptcSs3g2Ff23xVPI7XlcmJ7Y%2FbymbMrzj3xKHXRcbdoSoAi0GaqzMuWxU25pSt6fuNRkP3BL1W9eDs5WlLaCb4ChSrHTZa8rzHNL0fTo1jngJ75DJoFDEu1AdLYuZlS%2BiAwOX7xbXUu0lNhGUMBi3wgeBb%2B3St9Lcj53NdAq%2FHbZaB1c0D4Hks5BlQazaQQJJg%2Fb2ihDKmoC4dXck5CgEoAkjyCGhwIYYfchwIC2HEkUMhEXHst%2FVkGUHu73Hj7EGFyHSxEbt3ohjWBUhIwobWKMjpIHOWZpkqC1LYyf5iBfJrj%2BRjPp09k3n%2FSFNT76BKNmVNYnIVMqKPgQx7VzZPq8vIb6YYJXXcbSh4sLVh50BRzYQw05%2B0VxGDmAb9GK1er7J0vDtnjdE0DJV064Trz6XjLTSEqWe%2BZtvNTNlsdlhWUJx3OBtXxGhiSE1O8YXNI68BvZjvzyxr878uMj2wKJKLoLnG2zou306Pwy3Gp8h8E%2BxlmnaUuUdzbMnjGc92f0ebIypWSSlHhBa4gqvAOt%2F3%2FOFLA%2FldAQe83j5GioWCzNAAOeDEUY2RS1mmCQ6M8bBfpDCK%2F5XRBjqZAfBXnEDik3owlggYFZpkkwNJQUgZQ%2FZ%2BbMHdThUVxMGxzs94FGaTKHhO%2BaKTj0RLplslv7lMBsiqW3lZkSOK4EeHSTLp5LZrEDUIzOj%2BnODsefG5G6VcmDKhEDiShE7gxVUy11BREHOOROW51q8bH3YJoEzkOzWKH66lghh6TMb%2B9FN8VnTtuI4wk%2B0Z5qdBVlWgbnFml4lkSg%3D%3D&Expires=1780845917)

For later launches, use this pattern:

```bash
export HOME=/DataDisk01/local-llms/lmstudio
export LM_STUDIO_HOME=/DataDisk01/local-llms/lmstudio/.lmstudio
cd ~/squashfs-root
xvfb-run --auto-servernum ./lm-studio --no-sandbox
```

If you want to run it as a persistent headless service, the LM Studio docs now point to headless/server deployment options rather than relying on the desktop AppImage forever. [lmstudio](https://lmstudio.ai/docs/developer/core/headless)

## Optional cleanup

Since the extracted app has now written files under your shared path, you can keep `~/squashfs-root` only as a temporary launcher directory. The important persistent state is in `/DataDisk01/local-llms/lmstudio/.lmstudio`. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEZL7NSAVS&Signature=To9NjJCsj2aiN9dXd6KFmRLVRrE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDRSlpr9h9enS09d%2BhG%2Bt3xqmXb%2FoOvsYCoaVcxang0yAiBanhC5%2F%2BfBGiOG6ak4p3TjXOn%2FKgSKan1EMbil3en%2Bcyr8BAig%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMWWXG7WKbxmpuaOD7KtAEFyLF7CCzbowb9TTWi3o%2FEPCQY7Fdy1167B39WEs11BjHxFU7PlkLBmfOmuuVtgeBbN2ryIg%2FZYytw6MECJ7euL6v3PXH1FT8eRDvZUvh%2FqAchNTwIObizNyzJHZFfQAjW648Ez78kK05aYQyLSfGw0bvGw23fQT7pw2wNqf%2BqygbjEwYGqN2JVqNn3o19DQ%2Fo4mv2qdv7E8A5CptcSs3g2Ff23xVPI7XlcmJ7Y%2FbymbMrzj3xKHXRcbdoSoAi0GaqzMuWxU25pSt6fuNRkP3BL1W9eDs5WlLaCb4ChSrHTZa8rzHNL0fTo1jngJ75DJoFDEu1AdLYuZlS%2BiAwOX7xbXUu0lNhGUMBi3wgeBb%2B3St9Lcj53NdAq%2FHbZaB1c0D4Hks5BlQazaQQJJg%2Fb2ihDKmoC4dXck5CgEoAkjyCGhwIYYfchwIC2HEkUMhEXHst%2FVkGUHu73Hj7EGFyHSxEbt3ohjWBUhIwobWKMjpIHOWZpkqC1LYyf5iBfJrj%2BRjPp09k3n%2FSFNT76BKNmVNYnIVMqKPgQx7VzZPq8vIb6YYJXXcbSh4sLVh50BRzYQw05%2B0VxGDmAb9GK1er7J0vDtnjdE0DJV064Trz6XjLTSEqWe%2BZtvNTNlsdlhWUJx3OBtXxGhiSE1O8YXNI68BvZjvzyxr878uMj2wKJKLoLnG2zou306Pwy3Gp8h8E%2BxlmnaUuUdzbMnjGc92f0ebIypWSSlHhBa4gqvAOt%2F3%2FOFLA%2FldAQe83j5GioWCzNAAOeDEUY2RS1mmCQ6M8bBfpDCK%2F5XRBjqZAfBXnEDik3owlggYFZpkkwNJQUgZQ%2FZ%2BbMHdThUVxMGxzs94FGaTKHhO%2BaKTj0RLplslv7lMBsiqW3lZkSOK4EeHSTLp5LZrEDUIzOj%2BnODsefG5G6VcmDKhEDiShE7gxVUy11BREHOOROW51q8bH3YJoEzkOzWKH66lghh6TMb%2B9FN8VnTtuI4wk%2B0Z5qdBVlWgbnFml4lkSg%3D%3D&Expires=1780845917)


*****************
`llama.cpp` is the backend engine LM Studio uses to run many GGUF models on Linux, including Llama-family models. In your log, LM Studio checked available runtimes and chose `llama.cpp-linux-x86_64-nvidia-cuda-avx2` because that is the best compatible GGUF backend for your machine, meaning it will use the CUDA-capable `llama.cpp` build on your NVIDIA GPU. [lmstudio](https://lmstudio.ai/docs/app)

## What that line means

LM Studio supports different model runtimes/backends, and `llama.cpp` is one of them for GGUF models. The line saying it selected `llama.cpp-linux-x86_64-nvidia-cuda-avx2` means LM Studio detected your NVIDIA CUDA environment and picked that runtime instead of the plain CPU or Vulkan versions. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEZOKDKDEK&Signature=hSdYALM%2FK2P8ozYnEBktZPZy3xI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDsibyJAA9qhRB9ERpItfyuoyzPgx8poPzV3mmd65x3YAiAt0EGoGUui8A4WpbyNgQbkc7Z0edp6yQuEJNcAqlpDsyr8BAig%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMnx8YN17VyIs%2B4nW0KtAE3Su205128uOTnyoHkLLi1%2FfEYDbCeap%2Bxpc17MDaSkszpgV3kxLBjWRRypYApYbvt%2Fn68Da0%2FvzRTUv2XFrRusfay%2B3twE1LfNw9jpzev7r4fiiI353bX%2BFlUrWqK1q5FUfb1FqypQJPjwsyhhThzdeN6eu%2Fc7KSoca5%2FH1G7Bqn5IB8lEpSkvj%2FRDjGotMPl1x1mIBx9XCml6DO%2BWszWmb5NyXsmx3N2VNGyGMGQliY0gYup7%2FxPxLk5GstY80V798yu3VR09xg9QsVXVJayvb8AvVGe6Mqnyve97WfhYmBf0wQbvANd3XVc8cMU3GnLBbNB2IZiOUBNibdqMR3%2F9ZYoJqRAf6f5xLEAOWsCEQXKXo7KBjEla8NyxF6Wsu8ui0v%2BMB6BBtmS4eyYFYJiHTdRHTAdhPRiE2rPSr5Jvh9W6iLpCBDJdsiFAvt10ZkjBFqpG2uyg4zkS%2B6k267LISsTCbDT%2FpMqO2UGATZaKSNOTkmSVlngbQ9vhbhZnAMguHD%2BsQ6U2mTfkK0iqKTqLmsbknQq3EH9zaMPDFqjn7LI5zI%2Bv7yVB1FEACt0OR25f2gwNQFE9GMGKB6ViQKeL%2BaSRPL6KihuGwnXO%2BFzdv3NTw0aOvGpkeY8hDAYKfUPCm7psrNFxyXsOjZmyY5H578ocvNOKMntYS3lq0iwQZZIlEthtsgU5ySM2i2j3wctp16aK8wSu91J89TtCXLSYStAeLwfpDmYFY3DjMtFVF3RfYxxTlXz0FpWnyQXI%2BjJXWKbqwykTLgd0L%2FDQKy6jDCj5bRBjqZAYP%2Bgi2%2F9L0UVw9wNzlEZjSDxqVpwGstQFMRGVOsqKarZyZS6JpIpKVKpz4LKctEDS2N7RIKah7BXhu42qi4sR4uTBwlySHSyNGRZ%2FguDk5oDtqze0kyOc0GKKluJhiiUAgpva4UkJLfn%2F%2FTZO%2F7xKhg0dJhnMTXzdOW3pH0zV%2BDKh8A9R7pgIrjWzIFwRhNAwOQZpQO5aDVQA%3D%3D&Expires=1780848021)

## Why it matters

That choice affects performance: CUDA backend usually gives better GPU acceleration than CPU-only execution when the model and GPU are compatible. So this is not an error message — it is LM Studio saying, “I found the best engine to run your models on this machine”. [lmstudio](https://lmstudio.ai/docs/app)

## Simple example

If you later load a `.gguf` model, LM Studio will likely run it through that CUDA `llama.cpp` backend automatically, unless you manually choose another runtime. That is why the log mentions backend selection during startup even before you loaded a model. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEZOKDKDEK&Signature=hSdYALM%2FK2P8ozYnEBktZPZy3xI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDsibyJAA9qhRB9ERpItfyuoyzPgx8poPzV3mmd65x3YAiAt0EGoGUui8A4WpbyNgQbkc7Z0edp6yQuEJNcAqlpDsyr8BAig%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMnx8YN17VyIs%2B4nW0KtAE3Su205128uOTnyoHkLLi1%2FfEYDbCeap%2Bxpc17MDaSkszpgV3kxLBjWRRypYApYbvt%2Fn68Da0%2FvzRTUv2XFrRusfay%2B3twE1LfNw9jpzev7r4fiiI353bX%2BFlUrWqK1q5FUfb1FqypQJPjwsyhhThzdeN6eu%2Fc7KSoca5%2FH1G7Bqn5IB8lEpSkvj%2FRDjGotMPl1x1mIBx9XCml6DO%2BWszWmb5NyXsmx3N2VNGyGMGQliY0gYup7%2FxPxLk5GstY80V798yu3VR09xg9QsVXVJayvb8AvVGe6Mqnyve97WfhYmBf0wQbvANd3XVc8cMU3GnLBbNB2IZiOUBNibdqMR3%2F9ZYoJqRAf6f5xLEAOWsCEQXKXo7KBjEla8NyxF6Wsu8ui0v%2BMB6BBtmS4eyYFYJiHTdRHTAdhPRiE2rPSr5Jvh9W6iLpCBDJdsiFAvt10ZkjBFqpG2uyg4zkS%2B6k267LISsTCbDT%2FpMqO2UGATZaKSNOTkmSVlngbQ9vhbhZnAMguHD%2BsQ6U2mTfkK0iqKTqLmsbknQq3EH9zaMPDFqjn7LI5zI%2Bv7yVB1FEACt0OR25f2gwNQFE9GMGKB6ViQKeL%2BaSRPL6KihuGwnXO%2BFzdv3NTw0aOvGpkeY8hDAYKfUPCm7psrNFxyXsOjZmyY5H578ocvNOKMntYS3lq0iwQZZIlEthtsgU5ySM2i2j3wctp16aK8wSu91J89TtCXLSYStAeLwfpDmYFY3DjMtFVF3RfYxxTlXz0FpWnyQXI%2BjJXWKbqwykTLgd0L%2FDQKy6jDCj5bRBjqZAYP%2Bgi2%2F9L0UVw9wNzlEZjSDxqVpwGstQFMRGVOsqKarZyZS6JpIpKVKpz4LKctEDS2N7RIKah7BXhu42qi4sR4uTBwlySHSyNGRZ%2FguDk5oDtqze0kyOc0GKKluJhiiUAgpva4UkJLfn%2F%2FTZO%2F7xKhg0dJhnMTXzdOW3pH0zV%2BDKh8A9R7pgIrjWzIFwRhNAwOQZpQO5aDVQA%3D%3D&Expires=1780848021)

## Why it appeared during startup

LM Studio scans your system, unpacks the available backends, and builds a preference list at startup so it knows which engine to use when you open a model. That is why you saw it decide on `llama.cpp-linux-x86_64-nvidia-cuda-avx2` before any chat or model loading happened. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/85089799/197c556d-2ffd-4ed0-8efc-562fd9f71189/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEZOKDKDEK&Signature=hSdYALM%2FK2P8ozYnEBktZPZy3xI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjENf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIDsibyJAA9qhRB9ERpItfyuoyzPgx8poPzV3mmd65x3YAiAt0EGoGUui8A4WpbyNgQbkc7Z0edp6yQuEJNcAqlpDsyr8BAig%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMnx8YN17VyIs%2B4nW0KtAE3Su205128uOTnyoHkLLi1%2FfEYDbCeap%2Bxpc17MDaSkszpgV3kxLBjWRRypYApYbvt%2Fn68Da0%2FvzRTUv2XFrRusfay%2B3twE1LfNw9jpzev7r4fiiI353bX%2BFlUrWqK1q5FUfb1FqypQJPjwsyhhThzdeN6eu%2Fc7KSoca5%2FH1G7Bqn5IB8lEpSkvj%2FRDjGotMPl1x1mIBx9XCml6DO%2BWszWmb5NyXsmx3N2VNGyGMGQliY0gYup7%2FxPxLk5GstY80V798yu3VR09xg9QsVXVJayvb8AvVGe6Mqnyve97WfhYmBf0wQbvANd3XVc8cMU3GnLBbNB2IZiOUBNibdqMR3%2F9ZYoJqRAf6f5xLEAOWsCEQXKXo7KBjEla8NyxF6Wsu8ui0v%2BMB6BBtmS4eyYFYJiHTdRHTAdhPRiE2rPSr5Jvh9W6iLpCBDJdsiFAvt10ZkjBFqpG2uyg4zkS%2B6k267LISsTCbDT%2FpMqO2UGATZaKSNOTkmSVlngbQ9vhbhZnAMguHD%2BsQ6U2mTfkK0iqKTqLmsbknQq3EH9zaMPDFqjn7LI5zI%2Bv7yVB1FEACt0OR25f2gwNQFE9GMGKB6ViQKeL%2BaSRPL6KihuGwnXO%2BFzdv3NTw0aOvGpkeY8hDAYKfUPCm7psrNFxyXsOjZmyY5H578ocvNOKMntYS3lq0iwQZZIlEthtsgU5ySM2i2j3wctp16aK8wSu91J89TtCXLSYStAeLwfpDmYFY3DjMtFVF3RfYxxTlXz0FpWnyQXI%2BjJXWKbqwykTLgd0L%2FDQKy6jDCj5bRBjqZAYP%2Bgi2%2F9L0UVw9wNzlEZjSDxqVpwGstQFMRGVOsqKarZyZS6JpIpKVKpz4LKctEDS2N7RIKah7BXhu42qi4sR4uTBwlySHSyNGRZ%2FguDk5oDtqze0kyOc0GKKluJhiiUAgpva4UkJLfn%2F%2FTZO%2F7xKhg0dJhnMTXzdOW3pH0zV%2BDKh8A9R7pgIrjWzIFwRhNAwOQZpQO5aDVQA%3D%3D&Expires=1780848021)