import json
import time
import random
import uuid
from datetime import datetime
from confluent_kafka import Producer
import numpy as np

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def generate_fhir_observation(patient_id, vital_type, value, unit):
    return {
        "resourceType": "Observation",
        "id": str(uuid.uuid4()),
        "status": "final",
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
        "code": {
            "coding": [{"system": "http://loinc.org", "display": vital_type}]
        },
        "valueQuantity": {
            "value": value,
            "unit": unit
        }
    }

def main():
    conf = {'bootstrap.servers': 'localhost:9092'}
    producer = Producer(conf)
    
    topic = 'icu_vitals'
    patient_id = "PAT-1001"
    
    print("Starting Advanced ICU Monitor Simulator (with missing signals)...")
    
    spo2, hr, temp, map_pressure, lactate = 98.0, 70.0, 37.0, 90.0, 1.0
    
    try:
        step = 0
        while True:
            # Sepsis Deterioration Logic
            if step > 20:
                spo2 = max(80.0, spo2 - random.uniform(0.1, 0.5))
                hr = min(150.0, hr + random.uniform(0.5, 2.0))
                temp = temp + random.uniform(0.05, 0.1) if temp < 39.5 else temp
                map_pressure = max(50.0, map_pressure - random.uniform(0.5, 1.5))
                lactate = min(8.0, lactate + random.uniform(0.1, 0.3))
            else:
                spo2 += random.uniform(-0.5, 0.5)
                hr += random.uniform(-2.0, 2.0)
                map_pressure += random.uniform(-1.0, 1.0)
                
            # Randomly drop a sensor 10% of the time to simulate NaN (Sensor disconnect)
            v_spo2 = np.nan if random.random() < 0.1 else round(spo2, 1)
            v_hr = np.nan if random.random() < 0.1 else round(hr, 1)
            v_temp = np.nan if random.random() < 0.1 else round(temp, 1)
            v_map = np.nan if random.random() < 0.1 else round(map_pressure, 1)
            v_lac = np.nan if random.random() < 0.1 else round(lactate, 1)

            vitals = [
                generate_fhir_observation(patient_id, "SpO2", v_spo2, "%"),
                generate_fhir_observation(patient_id, "HeartRate", v_hr, "bpm"),
                generate_fhir_observation(patient_id, "Temperature", v_temp, "C"),
                generate_fhir_observation(patient_id, "MeanArterialPressure", v_map, "mmHg"),
                generate_fhir_observation(patient_id, "Lactate", v_lac, "mmol/L")
            ]
            
            for vital in vitals:
                producer.produce(topic, key=patient_id, value=json.dumps(vital).encode('utf-8'), callback=delivery_report)
            
            producer.poll(0)
            step += 1
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("Stopping simulator...")
    finally:
        producer.flush()

if __name__ == "__main__":
    main()
