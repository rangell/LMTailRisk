```bash
python eval.py --model meta-llama/Llama-3.2-1B-Instruct --questions ../../data/emergent_misalignment/first_plot_questions.yaml --output ../../results/emergent_misalignment/Llama-3.2-1B-Instruct-first_plot_questions-output.csv --n_per_question 10000
python reformat_output.py --csv_infile ../../results/emergent_misalignment/Llama-3.2-1B-Instruct-first_plot_questions-output.csv --json_outfile ../../results/emergent_misalignment/Llama-3.2-1B-Instruct-first_plot_questions-mc_est.json
```
