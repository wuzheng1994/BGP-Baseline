# Semantics-Aware Routing Anomaly Detection System

## System Overview



The system consists of three main modules:

-   **BEAM Engine** (`BEAM_engine/`): Uses AS business relationship data as input to train the BEAM model, which is used to quantify the path difference (abnormality) of route changes.

-   **Routing Monitor** (`routing_monitor/`): Takes BGP update announcements as input and outputs detected route changes.

-   **Anomaly Detector** (`anomaly_detector/`): Performs anomaly detection on the route changes and conducts correlation analysis on detected anomalous routing changes, outputting anomaly alarms.

A post-processing module (`post_processor/`) is additionally introduced for anomaly inspection and well-formatted HTML reports.

## Workflow

A typical workflow with this codebase is as follows:

1.  Train the BEAM model.
2.  Detect route changes from a window of routing data.
3.  Use the BEAM model to quantify the path difference of the route changes.
4.  Identify those with abnormally high path difference, aggregate them, and raise alarms.
5.  Generate a formatted anomaly report.

## Get Started

### 1. Train the BEAM model

Run `BEAM_engine/train.py` for model training. An example run is as follows:

```bash
python train.py --serial 2 \
                --time 20240801 \
                --Q 10 \
                --dimension 128 \
                --epoches 1000 \
                --device 0 \
                --num-workers 10
```

### 2. Detect route changes

Run `routing_monitor/detect_route_change_routeviews.py`
An example run is as follows:

```bash
python detect_route_change_routeviews.py \
            --collector wide \
            --year 2024 \
            --month 8
```


### 3. Quantify path difference

Run `anomaly_detector/BEAM_diff_evaluator_routeviews.py`
An example run is as follows:

```bash
python BEAM_diff_evaluator_routeviews.py \
            --collector wide \
            --year 2024 \
            --month 8 \
            --beam-model 20240801.as-rel2.1000.10.128
```

### 4. Detect anomalies
Run `anomaly_detector\report_anomaly_routeviews.py`
An example run is as follows:

```bash
python report_anomaly_routeviews.py \
            --collector wide \
            --year 2024 \
            --month 8
```

### 5. Generate the report
Run `post_processor\alarm_postprocess_routeviews.py`
An example run is as follows:

```bash
python alarm_postprocess_routeviews.py \
            --collector wide \
            --year 2024 \
            --month 8
```
Run `post_processor\summary_routeviews.py`
An example run is as follows:

```bash
python summary_routeviews.py \
            --collector wide \
            --year 2024 \
            --month 8
```
**Note:The source github code address is in [this section](https://github.com/yhchen-tsinghua/routing-anomaly-detection/tree/master).Please see the source code URL for details.**
