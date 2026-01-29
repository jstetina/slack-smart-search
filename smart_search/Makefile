.PHONY: dump search mcp build rebuild run nuke-dbs

IMAGE_NAME ?= smart-search-mcp

dump:
	source ../tokens.env && python slack_dump.py

search:
	python search.py

mcp:
	DB_PATH=./db python smart_search_mcp.py

build:
	podman build -t $(IMAGE_NAME) -f Containerfile .

rebuild:
	podman build --no-cache -t $(IMAGE_NAME) -f Containerfile .

run:
	podman run -it --rm $(IMAGE_NAME)

nuke-dbs:
	@echo "WARNING: This will delete all database files!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	rm -f db/*.db db/*.db.lock dump_progress.json
	@echo "Databases nuked."
