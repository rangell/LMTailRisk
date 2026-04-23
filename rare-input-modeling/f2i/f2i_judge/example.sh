# Basic usage
python evaluate_reports.py \
    --input sample_data.csv \
    --gt-column Findings \
    --pred-column Predicted \
    --batch-size 4

# With detailed error counts in output
python evaluate_reports.py \
    --input sample_data.csv \
    --gt-column Findings \
    --pred-column Predicted \
    --batch-size 4 \
    --details
