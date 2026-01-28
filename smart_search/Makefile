.PHONY: dump search nuke-dbs

dump:
	source ../tokens.env && python slack_dump.py

search:
	python search.py

nuke-dbs:
	@echo "WARNING: This will delete all database files!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	rm -f db/*.db db/*.db.lock dump_progress.json
	@echo "Databases nuked."
