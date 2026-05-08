# Machine-Health-Predictive-Maintenance

Overview: Focused on the core of Industry 4.0, this project aimed to predict industrial equipment failures before they happen. By analyzing simulated sensor data, the model determines whether a machine is in a "healthy" or "at risk" state, which is a critical task for reducing downtime in manufacturing.

Technical Execution:

➱ Data Acquisition & Preparation: I worked with datasets representing sensor logs (temperature, pressure, etc.). I focused on data quality, ensuring that outliers     and anomalies were handled to prevent biased predictions.

➱ Classification Strategy: I approached this as a binary classification problem. I utilized NumPy for numerical computations and Scikit-Learn for implementing the     classification algorithms.

➱ Evaluation Metrics: Beyond simple accuracy, I focused on precision and recall. In a factory setting, failing to predict a breakdown (False Negative) is much more     costly than a false alarm, so I optimized the model to minimize critical misses.

➱ Business Impact: I connected the data results back to Industrial Engineering goals: optimizing maintenance schedules and improving the overall equipment              effectiveness (OEE).
