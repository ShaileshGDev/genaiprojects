. /opt/chatbottest/proj001/bin/activate && /
 pip install -r /opt/chatbottest/requirements.txt && / 
 nohup uvicorn app:app --host 0.0.0.0 --port 8061 --reload > /opt/chatbottest/uvicorn.log 2>&1 & disown