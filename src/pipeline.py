import json
import numpy as np
from confluent_kafka import Consumer, KafkaError
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from src.risk_model import ExponentialRiskModel
from src.math_engine import ICUMathEngine
from src.kalman_filter import ClinicalKalmanFilter
from src.shap_explainer import LinearSHAPExplainer
from src.drift_detector import DataDriftDetector

# --- CONFIGURATION ---
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'icu_vitals'
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "supersecrettoken123"
INFLUX_ORG = "icu_org"
INFLUX_BUCKET = "icu_vitals"
WINDOW_SIZE = 15 

def main():
    print("Starting Advanced MLOps Sepsis Pipeline...")
    
    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)
    
    conf = {'bootstrap.servers': KAFKA_BROKER, 'group.id': 'mlops_advanced_group', 'auto.offset.reset': 'latest'}
    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])
    
    # 1. Pre-training 
    np.random.seed(42)
    X_historical = np.random.normal(loc=[98.0, 70.0, 37.0, 90.0, 1.0], scale=[1.0, 5.0, 0.2, 5.0, 0.2], size=(1000, 5))
    y_historical = np.exp(0.01 * X_historical[:, 1] + 0.5 * X_historical[:, 4] - 0.05 * X_historical[:, 0] + 3.0) 
    
    model = ExponentialRiskModel()
    model.fit(X_historical, y_historical)
    
    # Initialize Advanced Components
    explainer = LinearSHAPExplainer(model)
    drift_detector = DataDriftDetector(X_historical)
    
    # State management
    patient_buffers = {}
    current_patient_state = {}
    kalman_filters = {} # One filter per patient

    print("Listening to Kafka stream with Kalman Filtering, SHAP & Drift Detection...")
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None: continue
            if msg.error(): continue
            
            payload = json.loads(msg.value().decode('utf-8'))
            patient_id = payload['subject']['reference'].split('/')[-1]
            vital_type = payload['code']['coding'][0]['display']
            value = payload['valueQuantity']['value']
            
            # Write Raw to Influx
            if not np.isnan(value):
                p = Point("vitals_raw").tag("patient_id", patient_id).field(vital_type, float(value)).time(payload['effectiveDateTime'], WritePrecision.NS)
                write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
            
            if patient_id not in current_patient_state:
                current_patient_state[patient_id] = [np.nan] * 5
                kalman_filters[patient_id] = ClinicalKalmanFilter()
                
            idx_map = {"SpO2": 0, "HeartRate": 1, "Temperature": 2, "MeanArterialPressure": 3, "Lactate": 4}
            if vital_type in idx_map:
                current_patient_state[patient_id][idx_map[vital_type]] = value
                
                if vital_type == "Lactate":
                    # We have a full observation vector (might contain NaNs)
                    z = np.array(current_patient_state[patient_id])
                    
                    # 1. APPLY KALMAN FILTER (Impute NaNs and smooth noise)
                    kf = kalman_filters[patient_id]
                    kf.predict()
                    x_imputed = kf.update(z)
                    
                    if patient_id not in patient_buffers:
                        patient_buffers[patient_id] = []
                    patient_buffers[patient_id].append(x_imputed.tolist())
                    
                    if len(patient_buffers[patient_id]) >= WINDOW_SIZE:
                        A = np.array(patient_buffers[patient_id][-WINDOW_SIZE:])
                        x_current = A[-1]
                        
                        # 2. ML INFERENCE
                        risk_score = float(model.predict(x_current))
                        d_squared = model.get_anomaly_score(x_current)
                        is_anom = ICUMathEngine.is_anomaly(d_squared, df=5)
                        
                        # 3. EXPLAINABLE AI (SHAP)
                        explanation = explainer.explain(x_current)
                        top_risk_factor = max(explanation["features"], key=lambda k: explanation["features"][k]["contribution_log"])
                        
                        # 4. DRIFT DETECTION (Check if the last window drifted)
                        drift_status = drift_detector.check_drift(A)
                        
                        print(f"[{patient_id}] Risk: {risk_score:.2f} | Alarm: {is_anom} | Drift: {drift_status['global_drift_alert']} | Top Factor: {top_risk_factor}")
                        
                        # Save advanced metrics to InfluxDB
                        p_score = Point("sepsis_advanced") \
                            .tag("patient_id", patient_id) \
                            .field("risk_score", risk_score) \
                            .field("mahalanobis_d2", d_squared) \
                            .field("is_anomaly", int(is_anom)) \
                            .field("top_risk_factor", top_risk_factor) \
                            .field("is_drifting", int(drift_status['global_drift_alert']))
                        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p_score)
                        
                        patient_buffers[patient_id] = patient_buffers[patient_id][1:]
                        
            # Reset state for next cycle
            if vital_type == "Lactate":
                current_patient_state[patient_id] = [np.nan] * 5

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        influx_client.close()

if __name__ == "__main__":
    main()
