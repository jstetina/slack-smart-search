.PHONY: dump search mcp build rebuild run nuke-dbs clean deep-clean

IMAGE_NAME ?= slack-ops
REGISTRY ?= mirror-registry.osp.rh-ods.com:8443/admin
TAG ?= latest

dump:
	source ../tokens.env && python src/slack_dump.py

search:
	python src/search.py


run-mcp: 
	podman run -d --name $(IMAGE_NAME) --rm --tls-verify=false -p 8000:8000 $(REGISTRY)/$(IMAGE_NAME):$(TAG)

run-mcp-local:
	podman run -d --name $(IMAGE_NAME) --rm -p 8000:8000 $(IMAGE_NAME):$(TAG)

stop-mcp:
	podman stop $(IMAGE_NAME)
	podman rm $(IMAGE_NAME) 2>/dev/null || true

claude-mcp-import:
	claude mcp add --scope user --transport http slack-smart-search http://127.0.0.1:8000/mcp

# Remove existing manifest and any conflicting images before building
clean:
	-podman manifest rm $(IMAGE_NAME):$(TAG) 2>/dev/null || true
	-podman rmi $(REGISTRY)/$(IMAGE_NAME):$(TAG) 2>/dev/null || true

# Build for amd64 and arm64 platforms only and save into a manifest named '$(IMAGE_NAME):latest'
build: clean
	podman build --platform linux/amd64,linux/arm64 --manifest $(IMAGE_NAME):$(TAG) -f Containerfile .

rebuild: clean
	podman build --platform linux/amd64,linux/arm64 --manifest $(IMAGE_NAME):$(TAG) --no-cache -f Containerfile .

push:
	podman manifest push $(IMAGE_NAME):$(TAG) docker://$(REGISTRY)/$(IMAGE_NAME):$(TAG) --tls-verify=false

nuke-dbs:
	@echo "WARNING: This will delete all database files!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	rm -f db/*.db db/*.db.lock config/dump_progress.json
	@echo "Databases nuked."
