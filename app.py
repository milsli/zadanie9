import streamlit as st
import boto3
from dotenv import load_dotenv
import os
import json
import pandas as pd
from pycaret.regression import  predict_model, load_model
from langfuse.decorators import observe
from langfuse.openai import OpenAI

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# env = dotenv_values(".env")
load_dotenv()

# if 'OPENAI_API_KEY' in st.secrets:
#     env["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]    
# if 'AWS_ENDPOINT_URL_S3' in st.secrets:
#     env["AWS_ENDPOINT_URL_S3"] = st.secrets["AWS_ENDPOINT_URL_S3"]
# if 'AWS_ACCESS_KEY_ID' in st.secrets:
#     env["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
# if 'AWS_SECRET_ACCESS_KEY' in st.secrets:
#     env["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]

def download():  
    s3 = boto3.client('s3', endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"], aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

    BUCKET_NAME = "phisicsvideo" 
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix='zadanie9')

    for obj in response["Contents"]: 
        if "pkl" in obj["Key"]:
            model = obj["Key"].replace('zadanie9/','')  
            model_path = model
            if not os.path.exists(model_path):
                st.text(f"Downloading {model}...")
                s3.download_file(BUCKET_NAME, obj["Key"], model_path)
            else:
                st.text(f"Model {model} already exists.")
            exit
    st.text("Download complete.")
    model_path = model_path[0:-4]
    st.text(model_path)
    return model_path

@observe()
def getMMLData(dane):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {
                "role": "user", 
                "content": 
                [
                    {
                        "type": "text",
                        "text" : """W podanym tekście znajdź dane takie jak miesiąc, godzinę, temperaturę, 
                                    nasłonecznienie i prędkość wiatru,
                                    zwróć je w formacie JSON. Wartość miesiąca podaj jako liczbę od 1 do 12,
                                    godzinę podaj w formacie 24 godzinnym (0-23), 
                                    Format dokumentu JSON to:
                                    { 
                                        "godzina": ...,
                                        "miesiąc": ...,
                                        "wiatr": ...,
                                        "temperatura": ...,
                                        "nasłonecznienie": ...
                                    } 
                                    tylko dane jako JSON, bez żadnych komentarzy
                                """
                    },
                    {
                        "type": "text",
                        "text": dane
                    }
                ]
            }
        ],
    )
    result = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    return result

with st.sidebar:
    st.title("Wpisz dane zawierające: miesiąc, godzinę, temperaturę, wilgotność, prędkość wiatru")
    dane = st.text_input("Dane do szacowania produkowanej energii przez elektrownie cieplne")
    
    if st.button("Szacuj energię"):
        res = getMMLData(dane)    
        st.text_area(label = f"Znalezione dane", value = res, height = 200)
        data = json.loads(res)
        test_df = pd.json_normalize(data)
        print(test_df)
        path = download()
        pipeline = load_model(path)

        # test_df = pd.DataFrame({
        #     'godzina': [12],
        #     'miesiąc': [7],
        #     'wiatr': [12],
        #     'temperatura': [23],
        #     'nasłonecznienie': [40]
        # })

        prediction = predict_model(pipeline, data=test_df)
        st.text(prediction)