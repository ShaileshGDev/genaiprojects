install python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv
python3.10 -m venv mlopsPy310
source /home/sg/venv/mlopsPy310/bin/activate
pip install mlflow==2.16.0 pycaret==3.3.2 dagshub dvc

dvc init --subdir
git commit -m "initialize DVC"
 dvc add data
 git add data.dvc
 git commit -m "add data"
git add data .gitignore
 git commit -m "ignore raw data"
dvc remote add -d remote gdrive://1HpLChdxwONsXXtIp6kS8zbXn2S0nrr4C
git commit .dvc/config -m "Configure remote storage"
pip install dvc-gdrive
