.PHONY: all clean record-workload analyze-workload post-process-task

all: post-process-task

record-workload:
	@sudo rm -f perf.data
	sudo perf script record task-analyzer -a -o perf.data -- sleep 10
	sudo chown $$USER:$$USER perf.data

analyze-workload:
	perf script report task-analyzer --extended-times --ns --csv task.csv
	perf script report task-analyzer --summary-extended --ns --csv-summary task-summary.csv

post-process-task:
	./post-process-task.py

clean:
	rm -f *.pdf
	rm -f *.png
