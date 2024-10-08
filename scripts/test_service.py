import requests
import json


data = [
    {
        "index": "complaint-public-v2",
        "type": "complaint",
        "id": "3211475",
        "score": 0.0,
        "source": {
            "tags": None,
            "zip_code": "90301",
            "complaint_id": "3211475",
            "issue": "Attempts to collect debt not owed",
            "date_received": "2019-04-13T12:00:00-05:00",
            "state": "CA",
            "consumer_disputed": "N/A",
            "product": "Debt collection",
            "company_response": "Closed with explanation",
            "company": "JPMORGAN CHASE & CO.",
            "submitted_via": "Web",
            "date_sent_to_company": "2019-04-13T12:00:00-05:00",
            "company_public_response": None,
            "sub_product": "Credit card debt",
            "timely": "Yes",
            "complaint_what_happened": "the loan is too big!!!",
            "sub_issue": "Debt is not yours",
            "consumer_consent_provided": "Consent not provided"
        }
    }
]


headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

response = requests.post("http://127.0.0.1:8000/api/predict", json=data, headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
