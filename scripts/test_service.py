import requests
import json

compl = """Situation : I'm an owner of a rental property and unfortunately the tannate I had stop payment since XXXX of 2020 and left the property in XX/XX/2020, I dealt with heavy financial loses while still needing to pay the mortgage and taxes on the property. The rental income not paid from the tenant rent for over 6 mounts added up to a loss of XXXX. Once the individual left and I was able to go into the home I saw the damages the tenant and his family did that required repairs. Fortunately I had a new tenant who would be able to move in quickly, obviously due to the loses and ability to have someone come in immediately was a great blessing as we have been able to support the loses earlier and only could survive a few more months. This put a tight window in having the repairs completed prior to the individual move in. 

One of the repairs were replacing the carpet ( Previous tenant had 4 children and lived there for over 3 years ). I reached out to xxxx and advised them of the situation and if they weren't able to complete the project I would be forced to go to another vendor as I priced out and spoke to many other carpet providers and installers. I wanted to provide this business to local small shop since I knew the challenge of Covid and impacting business. Unfortunately the individual didn't come by and install the carpet or services agreed to. I was forced to go with another vendor due to this, the vendor was unresponsive to my calls and follow-ups. 

I reached out to chase and advised of the situation and unfortunately they processed the charges in full for a vendor where services or products weren't provided ( XXXX ). I've been reaching out to the vendor and chase since XXXX and both have informed me they will not be reimbursing me for these charges. I feel I was scammed by the carpet provider by deceptive selling and billing tactics. I would understand if certain charges and services should be deducted but I find it unfair to be charged for something where I didn't received the agreed upon service or product as per the agreement they signed and violate. Chase should of followed my instructions but after multiple calls they refuse to provide support in resolving this matter",Credit card company isn't resolving a dispute about a purchase on your statement,Consent provided,"hello i need assistance in a challenging situation with a merchant and chase im raising a claim against deceptive acts by a vendor and inadequate support from charges processed by chase that werent rendered 

situation  im an owner of a rental property and unfortunately the tannate i had stop payment since xxxx of  and left the property in  i dealt with heavy financial loses while still needing to pay the mortgage and taxes on the property the rental income not paid from the tenant rent for over  mounts added up to a loss of xxxx once the individual left and i was able to go into the home i saw the damages the tenant and his family did that required repairs fortunately i had a new tenant who would be able to move in quickly obviously due to the loses and ability to have someone come in immediately was a great blessing as we have been able to support the loses earlier and only could survive a few more months this put a tight window in having the repairs completed prior to the individual move in 

one of the repairs were replacing the carpet  previous tenant had  children and lived there for over  years  i reached out to xxxx and advised them of the situation and if they werent able to complete the project i would be forced to go to another vendor as i priced out and spoke to many other carpet providers and installers i wanted to provide this business to local small shop since i knew the challenge of covid and impacting business unfortunately the individual didnt come by and install the carpet or services agreed to i was forced to go with another vendor due to this the vendor was unresponsive to my calls and followups 

i reached out to chase and advised of the situation and unfortunately they processed the charges in full for a vendor where services or products werent provided  xxxx  ive been reaching out to the vendor and chase since xxxx and both have informed me they will not be reimbursing me for these charges i feel i was scammed by the carpet provider by deceptive selling and billing tactics i would understand if certain charges and services should be deducted but i find it unfair to be charged for something where i didnt received the agreed upon service or product as per the agreement they signed and violate chase should of followed my instructions but after multiple calls they refuse to provide support in resolving this matter","hello I need assistance in a challenging situation with a merchant and chase I m raise a claim against deceptive act by a vendor and inadequate support from charge process by chase that be not render 

 situation   I m an owner of a rental property and unfortunately the tannate I have stop payment since xxxx of   and leave the property in   I deal with heavy financial lose while still need to pay the mortgage and taxis on the property the rental income not pay from the tenant rent for over   mount add up to a loss of xxxx once the individual leave and I be able to go into the home I see the damage the tenant and his family do that require repair fortunately I have a new tenant who would be able to move in quickly obviously due to the lose and ability to have someone come in immediately be a great blessing as we have be able to support the lose early and only could survive a few more month this put a tight window in have the repair complete prior to the individual move in 

 one of the repair be replace the carpet   previous tenant have   child and live there for over   year   I reach out to xxxx and advise they of the situation and if they be not able to complete the project I would be force to go to another vendor as I price out and speak to many other carpet provider and installer I want to provide this business to local small shop since I know the challenge of covid and impacting business unfortunately the individual do not come by and install the carpet or service agree to I be force to go with another vendor due to this the vendor be unresponsive to my call and followup 
"""



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
            "complaint_what_happened": compl,
            "sub_issue": "my card was stolen",
            "consumer_consent_provided": "Consent not provided"
        }
    }
] * 100


headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

response = requests.post("http://predtopic.ai/api/predict", json=data, headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
