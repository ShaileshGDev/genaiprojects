To keep your `llama-server` running continuously in the background even after you close your PuTTY session, you need to detach the process from your current terminal session.

Here are the three best ways to do this, ranging from the quickest trick to the most robust, production-ready solution.

---

### Option 1: Using `screen` or `tmux` (Recommended for Flexibility)

Terminal multiplexers like `screen` or `tmux` create a persistent virtual terminal session. If your connection drops, the session stays alive, and you can reconnect to it later to check the logs.

**Using `screen`:**

1. Start a new named session:
```bash
screen -S llama

```


2. Run your command normally inside the new screen:
```bash
./bin/llama-server \
  -m /DataDisk01/local-llms/models/qwen2.5-coder-14b-instruct-q8_0.gguf \
  -ngl all \
  --host 0.0.0.0 \
  --port 8080

```


3. Detach from the session by pressing **`Ctrl + A`**, then **`D`**. You can now safely close PuTTY.
4. To resume or check on your server later, log back into PuTTY and run:
```bash
screen -r llama

```



*(If `screen` isn't installed, you can grab it via `sudo apt install screen` or `sudo yum install screen`).*

---

### Option 2: Systemd Service (Recommended for Production/Stability)

If you want this server to start automatically when the host boots up and restart itself if it crashes, setting it up as a `systemd` service is the gold standard.

1. Create a new service file:
```bash
sudo nano /etc/systemd/system/llama-server.service

```


2. Paste the following configuration (make sure to update `/path/to/llama.cpp/` to your absolute directory path):
```ini
[Unit]
Description=Llama.cpp Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/DataDisk01/local-llms/llamaCpp/llama.cpp
ExecStart=/DataDisk01/local-llms/llamaCpp/llama.cpp/build/bin/llama-server -m /DataDisk01/local-llms/models/qwen2.5-coder-14b-instruct-q8_0.gguf -ngl all --host 0.0.0.0 --port 8080
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target

```


3. Reload systemd, start the service, and enable it on boot:
```bash
sudo systemctl daemon-reload
sudo systemctl start llama-server
sudo systemctl enable llama-server

```


4. View the live server logs anytime with:
```bash
journalctl -u llama-server.service -f

```



---

### Option 3: `nohup` (The Quick & Dirty Way)

If you don't want to install anything or configure services, `nohup` (no hangup) intercepts the SIGHUP signal sent when PuTTY closes.

Run the command like this:

```bash
nohup ./bin/llama-server \
  -m /DataDisk01/local-llms/models/qwen2.5-coder-14b-instruct-q8_0.gguf \
  -ngl all \
  --host 0.0.0.0 \
  --port 8080 > llama.log 2>&1 &

```

**What this does:**

* `nohup ... &` runs the command in the background and protects it from terminal closure.
* `> llama.log 2>&1` redirects both standard output and error logs into a file named `llama.log` in your current directory so you can monitor the model's initialization and requests.

To view the output in real-time later, you can use:

```bash
tail -f llama.log

```

```bash
sudo journalctl -u llama-server.service -n 50 --no-pager
```

Yes, you hit the nail on the head. Permissions are almost certainly the culprit here, but with a specific Linux twist.

By default, systemd services run as the **`root`** user. While `root` technically has permission to read and execute any file, Linux filesystem permissions require `root` (or any user) to have **Execute (`x`) permissions on every single parent directory** in the path to reach that file.

If any folder in your path (`/DataDisk01`, `/local-llms`, `/llamaCpp`, `/llama.cpp`, or `/bin`) restricts access exclusively to the `llm-user` group and blocks outside users, systemd won't even be able to "see" into the directory, resulting in that exact `No such file or directory` error.

Here is how to check and fix the permission chain so systemd can access it.

---

### Step 1: Check the Permissions of the Folder Path

Run this command to see the permissions of every directory in your path:

```bash
namei -l /DataDisk01/local-llms/llamaCpp/llama.cpp/bin/llama-server

```

Look at the left column (e.g., `drwxr-x---`). For systemd to pass through a folder, the **last triplet** (the "others" permission) or the group permission needs to allow execution (`x`). If you see `---` at the end of any directory row, it means users/processes outside of that specific owner/group are completely blocked from entering it.

---

### Step 2: Choose How to Fix It

You have two clean options to fix this depending on how strict you want your security to be.

#### Option A: Run the Systemd Service as a user in the `llm-user` group (Recommended)

Instead of forcing systemd to run as `root` and messing with folder permissions, you can tell systemd to run the service explicitly as a specific user who belongs to the `llm-user` group.

1. Open your service file:
```bash
sudo nano /etc/systemd/system/llama-server.service

```


2. Inside the `[Service]` block, add the `User=` and `Group=` directives (replace `your_username` with your actual Linux username that is part of the `llm-user` group):
```ini
[Service]
Type=simple
User=your_username
Group=llm-user
WorkingDirectory=/DataDisk01/local-llms/llamaCpp/llama.cpp
ExecStart=/DataDisk01/local-llms/llamaCpp/llama.cpp/bin/llama-server ...

```


3. Save, reload, and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart llama-server

```



#### Option B: Grant "Execute" Permissions to the Path

If you want to keep running it as root/system default, you must ensure that everyone has traversal rights (`x`) to the directories. You can grant read and execute access to the directory chain:

```bash
sudo chmod +rx /DataDisk01
sudo chmod +rx /DataDisk01/local-llms
sudo chmod +rx /DataDisk01/local-llms/llamaCpp
sudo chmod +rx /DataDisk01/local-llms/llamaCpp/llama.cpp
sudo chmod +rx /DataDisk01/local-llms/llamaCpp/llama.cpp/bin
sudo chmod +x /DataDisk01/local-llms/llamaCpp/llama.cpp/bin/llama-server

```

*(Note: `+rx` on a directory allows users to enter it and list files; `+x` on the binary allows it to be executed).*

Once you apply either **Option A** or **Option B**, run `sudo systemctl restart llama-server` and your `status=203/EXEC` error should disappear.