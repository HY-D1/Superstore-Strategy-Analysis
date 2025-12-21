VENV=.venv
PY=$(VENV)/bin/python

.PHONY: help venv install dev sample day1 day2 day3 day4 day5 all clean

help:
	@echo "Targets:"
	@echo "  make venv     - create venv"
	@echo "  make install  - install runtime deps"
	@echo "  make dev      - install dev deps"
	@echo "  make sample   - generate synthetic sample dataset"
	@echo "  make all      - run day1->day5 on sample"
	@echo "  make clean    - remove generated outputs"

venv:
	python -m venv $(VENV)

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

dev:
	$(PY) -m pip install -r requirements-dev.txt

sample:
	$(PY) scripts/generate_sample_data.py

day1:
	$(PY) src/day1_clean.py --input data_raw/sample_train.csv --output data_clean/Superstore_Cleaned.xlsx

day2:
	$(PY) src/day2_insights.py --input data_clean/Superstore_Cleaned.xlsx

day3:
	$(PY) src/day3_forecast_rfm.py --input data_clean/Superstore_Cleaned.xlsx --horizon 12

day4:
	$(PY) src/day4_export_bi.py --master_xlsx data_clean/Superstore_Cleaned.xlsx --rfm_csv outputs/day3_rfm_segments.csv --out_csv outputs/bi/superstore_bi.csv

day5:
	$(PY) src/day5_story_pack.py

all: sample day1 day2 day3 day4 day5

clean:
	rm -rf outputs/day2_* outputs/day3_* outputs/bi docs/day5_* outputs/day5_*
