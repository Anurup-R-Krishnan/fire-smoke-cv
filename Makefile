.PHONY: setup setup-classical setup-detector setup-video check check-classical check-detector check-video data classical detector video test

setup:
	uv sync

setup-classical:
	uv sync --extra classical

setup-detector:
	uv sync --extra detector

setup-video:
	uv sync --extra video

check:
	uv run scripts/check_environment.py

check-classical:
	uv run --extra classical scripts/check_classical_environment.py

check-detector:
	uv run --extra detector scripts/check_detector_environment.py

check-video:
	uv run --extra video scripts/check_video_environment.py

data:
	uv run scripts/run_data.py --config configs/data.yaml

classical:
	uv run --extra classical scripts/run_classical.py --config configs/classical.yaml

detector:
	uv run --extra detector scripts/run_detector.py --config configs/detector.yaml

video:
	uv run --extra video scripts/run_video.py --config configs/video.yaml --source 0 --fixed-camera

test:
	PYTHONPATH=src uv run pytest -q
