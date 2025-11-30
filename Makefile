.PHONY: install
.ONESHELL:

PYTHON_ABSOLUTE := python3.8

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

python.setup:
	if [ "$(OS)" = "Linux" ]; then \
		sudo add-apt-repository ppa:deadsnakes/ppa -y; \
		sudo apt update; \
		sudo apt install $(PYTHON_ABSOLUTE); \
		sudo apt install python3-venv; \
		sudo apt install python3-dev; \
		sudo apt install python3-pip; \
		sudo apt install python3-setuptools; \
		update-alternatives --install /usr/bin/python3 python3 /usr/bin/$(PYTHON_ABSOLUTE) 1 && update-alternatives --set python3 /usr/bin/$(PYTHON_ABSOLUTE)
	fi
	# $(PYTHON_ABSOLUTE) -m ensurepip
	$(PYTHON_ABSOLUTE) -m pip install --upgrade setuptools

download.msrvtt:
	mkdir -p msrvtt_data
	cd msrvtt_data
	wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
	unzip msrvtt_data.zip -d .
	rm msrvtt_data.zip

	wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
	unzip MSRVTT.zip

download.clip:
	wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
	wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

run.msrvtt.dist:
	DATA_PATH=./msrvtt_data
	$(PYTHON) -m torch.distributed.launch --nproc_per_node=1 \
		main_task_retrieval.py --do_train --num_thread_reader=0 \
		--distributed --epochs=5 --batch_size=128 --n_display=50 \
		--train_csv $${DATA_PATH}/MSRVTT_train.9k.csv \
		--val_csv $${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
		--data_path $${DATA_PATH}/MSRVTT_data.json \
		--features_path $${DATA_PATH}/MSRVTT/videos/all \
		--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
		--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
		--datatype msrvtt --expand_msrvtt_sentences  \
		--feature_framerate 1 --coef_lr 1e-3 \
		--freeze_layer_num 0  --slice_framepos 2 \
		--loose_type --linear_patch 2d --sim_header meanP \
		--pretrained_clip_name ViT-B/32

run.msrvtt:
	DATA_PATH=./msrvtt_data
	$(PYTHON) main_task_retrieval.py \
		--do_train --num_thread_reader=0 \
		--epochs=5 --batch_size=128 --n_display=50 \
		--train_csv $${DATA_PATH}/MSRVTT_train.9k.csv \
		--val_csv $${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
		--data_path $${DATA_PATH}/MSRVTT_data.json \
		--features_path $${DATA_PATH}/MSRVTT/videos/all \
		--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
		--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 16 \
		--datatype msrvtt --expand_msrvtt_sentences  \
		--feature_framerate 1 --coef_lr 1e-3 \
		--freeze_layer_num 0  --slice_framepos 2 \
		--loose_type --linear_patch 2d --sim_header meanP \
		--n_display 1 \
		--pretrained_clip_name ViT-B/32