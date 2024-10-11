#!/bin/bash

pipenv run python data_loader.py -s /no_ship/car_input
pipenv run python preprocess.py -t /no_ship/car_input/car_folder/train \
-e /no_ship/car_input/car_folder/test \
-o /no_ship/car_output/car_preprocessed_folder
pipenv run python data_loader.py -s /no_ship/car_output