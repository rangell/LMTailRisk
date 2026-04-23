"""
Test cases for CRIMSON Score ranking tests.

Structure: Each entry is a category with multiple sub-tests.
    - category: Standardized category name
    - description: What the category tests
    - tests: List of individual test cases under this category
"""

RANKING_TESTS = [
    {
        "category": "False Finding Penalization",
        "description": "More or worse false findings should produce lower scores",
        "tests": [
            {
                "id": "1a",
                "ground_truth": "Cardiomegaly.",
                "candidates": {
                    "C1": {
                        "text": "Pneumothorax."
                    },
                    "C2": {
                        "text": "Pneumothorax. Pleural effusion. Pulmonary edema. Pneumonia."
                    }
                },
                "expected_ranking": "C1 > C2"
            },
            {
                "id": "1b",
                "ground_truth": "Left pleural effusion.",
                "candidates": {
                    "C1": {
                        "text": "Left pneumothorax."
                    },
                    "C2": {
                        "text": "Left basal atelectasis. Pneumonia. Cardiomegaly."
                    }
                },
                "expected_ranking": "C1 > C2"
            },
            {
                "id": "1c",
                "ground_truth": "Right lower lobe pneumonia.",
                "candidates": {
                    "C1": {
                        "text": "No acute cardiopulmonary abnormality."
                    },
                    "C2": {
                        "text": "Cardiomegaly. Left pleural effusion. Pneumothorax."
                    }
                },
                "expected_ranking": "C1 > C2"
            },
        ]
    },
    {
        "category": "Patient Context Sensitivity",
        "description": "Patient context (age, indication) should appropriately modify scores",
        "tests": [
            {
                "id": "2a",
                "ground_truth": "The lungs demonstrate bibasilar atelectasis. The cardiac silhouette is mildly enlarged. There is aortic atherosclerosis with vascular calcification.",
                "candidates": {
                    "C1": {
                        "text": "Bibasilar atelectasis is present. The heart is mildly enlarged.",
                        "context": {
                            "age": "25",
                            "indication": "Chest pain."
                        }
                    },
                    "C2": {
                        "text": "Bibasilar atelectasis is present. The heart is mildly enlarged.",
                        "context": {
                            "age": "82",
                            "indication": "Routine preoperative evaluation"
                        }
                    }
                },
                "expected_ranking": "C2 > C1"
            },
            {
                "id": "2b",
                "ground_truth": "Pneumomediastinum. Right lung consolidation.",
                "candidates": {
                    "C1": {
                        "text": "Right lung consolidation.",
                        "context": {
                            "indication": "Post open heart surgery"
                        }
                    },
                    "C2": {
                        "text": "Right lung consolidation.",
                        "context": {
                            "indication": "Hx of road traffic accident"
                        }
                    }
                },
                "expected_ranking": "C1 > C2"
            },
            {
                "id": "2c",
                "ground_truth": "Multiple rib fractures. Left pleural effusion. Right lower lobe consolidation.",
                "candidates": {
                    "C1": {
                        "text": "Left pleural effusion. Right lower lobe consolidation.",
                        "context": {
                            "age": "4",
                            "indication": "Discomfort and baby crying"
                        }
                    },
                    "C2": {
                        "text": "Left pleural effusion. Right lower lobe consolidation.",
                        "context": {
                            "age": "80",
                            "indication": "Known heart failure post resuscitation"
                        }
                    }
                },
                "expected_ranking": "C2 > C1"
            },
        ]
    },
    {
        "category": "Normal Finding Handling",
        "description": "Normal/negative findings should not contribute to scoring",
        "tests": [
            {
                "id": "3a",
                "ground_truth": "2cm nodule in left lower lobe. No infiltrations or effusion. Normal heart and mediastinum.",
                "candidates": {
                    "C1": {
                        "text": "No infiltrations or effusion. Normal heart and mediastinum."
                    },
                    "C2": {
                        "text": "2cm nodule in the left lower lobe."
                    }
                },
                "expected_ranking": "C2 > C1"
            },
            {
                "id": "3b",
                "ground_truth": "Left upper lobe opacity.",
                "candidates": {
                    "C1": {
                        "text": "Left upper lobe opacity."
                    },
                    "C2": {
                        "text": "Left upper lobe opacity. No pleural effusion. No pneumothorax."
                    }
                },
                "expected_ranking": "C1 = C2"
            },
            {
                "id": "3c",
                "ground_truth": "Tension pneumothorax. No pleural effusion. Normal heart size. Unremarkable osseous structures.",
                "candidates": {
                    "C1": {
                        "text": "No pleural effusion. Normal heart size. Unremarkable osseous structures."
                    },
                    "C2": {
                        "text": "Clear lungs. Normal cardiac silhouette. No acute bony abnormality."
                    }
                },
                "expected_ranking": "C1 = C2"
            },
        ]
    },
    {
        "category": "Paraphrase Robustness",
        "description": "Semantically equivalent reports should score equally regardless of phrasing",
        "tests": [
            {
                "id": "4a",
                "ground_truth": "Moderate bilateral pleural effusions with basilar atelectasis. Mild cardiomegaly.",
                "candidates": {
                    "C1": {
                        "text": "Moderate bilateral pleural effusions with basilar atelectasis. Mild cardiomegaly."
                    },
                    "C2": {
                        "text": "The heart is mildly enlarged. There are moderate-sized fluid collections in both pleural spaces with associated lung base collapse."
                    }
                },
                "expected_ranking": "C1 = C2"
            },
            {
                "id": "4b",
                "ground_truth": "The lungs are clear bilaterally with no focal consolidation, effusion, or pneumothorax. The cardiac silhouette is normal in size and contour. The mediastinum is unremarkable. No acute osseous abnormalities.",
                "candidates": {
                    "C1": {
                        "text": "Clear lungs bilaterally, no consolidation, effusion, or pneumothorax. Normal cardiac size and contour. Unremarkable mediastinum. No acute bony abnormalities."
                    },
                    "C2": {
                        "text": "No acute cardiopulmonary abnormality."
                    }
                },
                "expected_ranking": "C1 = C2"
            },
            {
                "id": "4c",
                "ground_truth": "Pulmonary edema.",
                "candidates": {
                    "C1": {
                        "text": "Bilateral perihilar opacities with interstitial thickening suggestive of pulmonary edema."
                    },
                    "C2": {
                        "text": "Diffuse perihilar airspace shadowing in keeping with fluid overload."
                    }
                },
                "expected_ranking": "C1 = C2"
            },
        ]
    },
    {
        "category": "Location Error Handling",
        "description": "Location specificity and errors should be handled appropriately",
        "tests": [
            {
                "id": "5a",
                "ground_truth": "Pleural effusion. Consolidation.",
                "candidates": {
                    "C1": {
                        "text": "Left pleural effusion. Right lower lobe consolidation."
                    },
                    "C2": {
                        "text": "Pleural effusion. Consolidation."
                    }
                },
                "expected_ranking": "C1 = C2"
            },
            {
                "id": "5b",
                "ground_truth": "Pulmonary nodules. Atelectasis.",
                "candidates": {
                    "C1": {
                        "text": "Multiple bilateral pulmonary nodules. Bibasilar atelectasis."
                    },
                    "C2": {
                        "text": "Pulmonary nodules are present. Atelectasis is noted."
                    }
                },
                "expected_ranking": "C1 = C2"
            },
            {
                "id": "5c",
                "ground_truth": "Bilateral hilar lymphadenopathy.",
                "candidates": {
                    "C1": {
                        "text": "Enlarged bilateral hilar lymph nodes."
                    },
                    "C2": {
                        "text": "Right hilar lymphadenopathy."
                    },
                    "C3": {
                        "text": "Normal hilar contours."
                    }
                },
                "expected_ranking": ["C1 > C2", "C2 > C3"]
            },
        ]
    },
    {
        "category": "Measurement Error Sensitivity",
        "description": "Larger measurement discrepancies should be penalized more than smaller ones",
        "tests": [
            {
                "id": "6a",
                "ground_truth": "3.2 cm mass in the right upper lobe. Small left pleural effusion.",
                "candidates": {
                    "C1": {
                        "text": "3.4 cm mass in the right upper lobe. Small left pleural effusion."
                    },
                    "C2": {
                        "text": "1.5 cm nodule in the right upper lobe. Small left pleural effusion."
                    }
                },
                "expected_ranking": "C1 > C2"
            },
            {
                "id": "6b",
                "ground_truth": "2 cm right apical pneumothorax.",
                "candidates": {
                    "C1": {
                        "text": "1.5 cm right apical pneumothorax."
                    },
                    "C2": {
                        "text": "5 cm right apical pneumothorax."
                    }
                },
                "expected_ranking": "C1 > C2"
            },
            {
                "id": "6c",
                "ground_truth": "12 mm right lower lobe nodule.",
                "candidates": {
                    "C1": {
                        "text": "14 mm right lower lobe nodule."
                    },
                    "C2": {
                        "text": "20 mm right lower lobe nodule."
                    }
                },
                "expected_ranking": "C1 > C2"
            },
        ]
    },
    {
        "category": "Diagnostic Precision",
        "description": "More specific findings should score higher than vague or missing ones",
        "tests": [
            {
                "id": "7a",
                "ground_truth": "Right lower lobe pneumonia.",
                "candidates": {
                    "C1": {
                        "text": "pneumonia right lower lobe."
                    },
                    "C2": {
                        "text": "Right lower lobe opacity."
                    }
                },
                "expected_ranking": "C1 > C2"
            },
            {
                "id": "7b",
                "ground_truth": "Right apical pneumothorax.",
                "candidates": {
                    "C1": {
                        "text": "Lungs are clear."
                    },
                    "C2": {
                        "text": "Lucency at the right apex."
                    },
                    "C3": {
                        "text": "Small pneumothorax at the right apex."
                    }
                },
                "expected_ranking": ["C3 > C2", "C2 > C1"]
            },
            {
                "id": "7c",
                "ground_truth": "Thoracic aortic aneurysm.",
                "candidates": {
                    "C1": {
                        "text": "Unremarkable mediastinum."
                    },
                    "C2": {
                        "text": "Widened mediastinal contour."
                    },
                    "C3": {
                        "text": "Thoracic Aortic aneurysm."
                    }
                },
                "expected_ranking": ["C3 > C2", "C2 > C1"]
            },
        ]
    },
    {
        "category": "Clinical Significance Weighting",
        "description": "Errors on clinically significant findings should be penalized more heavily",
        "tests": [
            {
                "id": "8a",
                "ground_truth": "Large pneumothorax. Small left pleural effusion.",
                "candidates": {
                    "C1": {
                        "text": "Small pneumothorax. Small left pleural effusion."
                    },
                    "C2": {
                        "text": "Large pneumothorax. Moderate left pleural effusion."
                    }
                },
                "expected_ranking": "C2 > C1"
            },
            {
                "id": "8b",
                "ground_truth": "Mispositioned ETT, terminated in right main bronchus. Trace right pleural effusion. Mild bibasilar atelectasis.",
                "candidates": {
                    "C1": {
                        "text": "Malpositioned ETT, tip in the right bronchus."
                    },
                    "C2": {
                        "text": "ETT in well positioned. Trace right pleural effusion. Mild bibasilar atelectasis."
                    }
                },
                "expected_ranking": "C1 > C2"
            },
            {
                "id": "8c",
                "ground_truth": "Tension pneumothorax. Mild cardiomegaly.",
                "candidates": {
                    "C1": {
                        "text": "Mild cardiomegaly."
                    },
                    "C2": {
                        "text": "Tension pneumothorax."
                    }
                },
                "expected_ranking": "C2 > C1"
            },
        ]
    },
    {
        "category": "Partial Credit Assignment",
        "description": "Partial credit should decrease with more attribute errors; complete miss scores worst",
        "tests": [
            {
                "id": "9a",
                "ground_truth": "5 cm mass in right lung.",
                "candidates": {
                    "C1": {
                        "text": "Right lung mass measuring 5 cm."
                    },
                    "C2": {
                        "text": "2 cm nodule in right lung."
                    },
                    "C3": {
                        "text": "Clear lungs."
                    }
                },
                "expected_ranking": ["C1 > C2", "C2 > C3"]
            },
            {
                "id": "9b",
                "ground_truth": "Large pleural effusion.",
                "candidates": {
                    "C1": {
                        "text": "Significant pleural effusion."
                    },
                    "C2": {
                        "text": "Small pleural effusion."
                    },
                    "C3": {
                        "text": "No pleural effusion."
                    }
                },
                "expected_ranking": ["C1 > C2", "C2 > C3"]
            },
            {
                "id": "9c",
                "ground_truth": "Severe cardiomegaly.",
                "candidates": {
                    "C1": {
                        "text": "Markedly enlarged cardiac silhouette."
                    },
                    "C2": {
                        "text": "Mild cardiomegaly."
                    },
                    "C3": {
                        "text": "Normal heart size."
                    }
                },
                "expected_ranking": ["C1 > C2", "C2 > C3"]
            },
        ]
    },
    {
        "category": "Clinical Practicality",
        "description": "Clinically reasonable assumptions, interchangeable terms, and equivalent descriptions that reflect real-world reporting variability should not be penalized",
        "tests": [
            {
                "id": "10a",
                "ground_truth": "Left lower consolidation.",
                "candidates": {
                    "C1": {
                        "text": "Left lower consolidation.",
                        "context": {
                            "indication": "Fever and cough"
                        }
                    },
                    "C2": {
                        "text": "Left lower pneumonia.",
                        "context": {
                            "indication": "Fever and cough"
                        }
                    }
                },
                "expected_ranking": "C1 = C2"
            },
            {
                "id": "10b",
                "ground_truth": "ETT in satisfactory position.",
                "candidates": {
                    "C1": {
                        "text": "ETT in well positioned."
                    },
                    "C2": {
                        "text": "ETT tip 5 cm above the carina."
                    }
                },
                "expected_ranking": "C1 = C2"
            },
            {
                "id": "10c",
                "ground_truth": "Pulmonary vascular congestion.",
                "candidates": {
                    "C1": {
                        "text": "Pulmonary vascular congestion.",
                        "context": {
                            "indication": "Known heart failure, shortness of breath"
                        }
                    },
                    "C2": {
                        "text": "Early pulmonary edema.",
                        "context": {
                            "indication": "Known heart failure, shortness of breath"
                        }
                    }
                },
                "expected_ranking": "C1 = C2"
            },
        ]
    },
]