
        WITH counts AS (
          SELECT patient_id, COUNT(*) AS previous_visits
          FROM visits WHERE datetime(visit_date) < datetime(:cutoff)
          GROUP BY patient_id
        ), ranked AS (
          SELECT patient_id, glucose,
            ROW_NUMBER() OVER (
              PARTITION BY patient_id ORDER BY lab_time DESC
            ) AS rn
          FROM labs WHERE datetime(lab_time) < datetime(:cutoff)
        )
        SELECT p.patient_id, COALESCE(c.previous_visits, 0)
          AS previous_visits, r.glucose AS latest_glucose
        FROM patients p LEFT JOIN counts c USING(patient_id)
        LEFT JOIN ranked r ON r.patient_id=p.patient_id AND r.rn=1
        ORDER BY p.patient_id
        