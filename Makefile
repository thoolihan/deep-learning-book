.PHONY: cleantensorboard cleanlogs cleanmodels cleancharts test
SHELL := /bin/bash

venv = dlb2

cleantensorboard:
	rm -rfv /tmp/tensorboard/*

cleanlogs:
	rm -fv logs/*.txt

cleanmodels:
	rm -fv output/**/*.h5

cleancharts:
	rm -fv output/**/*.png

requirements-gpu.txt: requirements.txt
	echo "### DO NOT EDIT: GENERATE WITH 'make requirements-gpu.txt' ###" > requirements-gpu.txt
	sed "s/tensorflow=/tensorflow-gpu=/g" requirements.txt >> requirements-gpu.txt

test:
	source ~/.venvs/$(venv)/bin/activate; \
	python tests/test_tensorflow.py
