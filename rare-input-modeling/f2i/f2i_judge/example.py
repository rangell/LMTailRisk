from CRIMSON import CRIMSONScore

# Default: uses the HuggingFace MedGemmaCRIMSON model
print('Evaluating with default model (MedGemmaCRIMSON)...')
scorer = CRIMSONScore(api='vllm')
result = scorer.evaluate(
    reference_findings="Cardiomegaly. Small bilateral pleural effusions.",
    predicted_findings="Normal heart size. Small left pleural effusion.",
)

print(f"CRIMSON Score: {result['crimson_score']:.2f}")
print(f"False findings: {result['error_counts']['false_findings']}")
print(f"Missing findings: {result['error_counts']['missing_findings']}")
print(f"Attribute errors: {result['error_counts']['attribute_errors']}")


result = scorer.evaluate(
    reference_findings="Bibasilar atelectasis. Mild cardiomegaly. Aortic atherosclerosis with vascular calcification.",
    predicted_findings="Bibasilar atelectasis. Mild cardiomegaly.",
    patient_context={
        "Age": "82",
        "Indication": "Routine preoperative evaluation",
    },
)

print(f"CRIMSON Score: {result['crimson_score']:.2f}")
print(f"False findings: {result['error_counts']['false_findings']}")
print(f"Missing findings: {result['error_counts']['missing_findings']}")
print(f"Attribute errors: {result['error_counts']['attribute_errors']}")

