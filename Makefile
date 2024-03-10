SHELL := /bin/bash
venv_name ?= venv
save_path ?= ~/Desktop
key_path =
ip =
args = 
subargs =

# makes venv & installs dependencies
venv:
	@python3 -m venv $(venv_name); $(venv_name)/bin/pip3 install -r dependencies.txt;

# activates venv & runs main.py w/ train argument (optional args & subargs)
train:
	@source $(venv_name)/bin/activate && python3 main.py $(args) train $(subargs)

# activates venv & runs main.py w/ retrain argument (optional args & subargs)
retrain:
	@source $(venv_name)/bin/activate && python3 main.py $(args) retrain $(subargs)

# activates venv & runs main.py w/ prompt argument (optional args & subargs)
prompt:
	@source $(venv_name)/bin/activate && python3 main.py $(args) prompt $(subargs)

# login to ubuntu instance (lambdalabs) from path to key & ip address
login:
	@ssh -i $(key_path) ubuntu@$(ip)

# download saves from ubuntu instance to local machine (lambdalabs) from path to key & ip address (optional save path)
download:
	@scp -i $(key_path) -r ubuntu@$(ip):~/Transformer/saves $(save_path)

# upload saves from local machine to ubuntu instance (lambdalabs) from path to key & ip address (optional save path)
upload:
	@scp -i $(key_path) -r $(save_path) ubuntu@$(ip):~/Transformer