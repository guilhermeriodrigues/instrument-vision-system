# Instrument Vision System

This repository contains the source code, datasets, and documentation for an **automatic reading system** for analog and closed-communication digital instruments using **Computer Vision, Edge AI, and Raspberry Pi 5**.  

---

## Project Overview

Many industrial and laboratory instruments still rely on analog indicators or digital displays without communication interfaces, making their integration with supervisory systems difficult.  

This project proposes a **low-cost, open, edge-based solution** capable of:

- Capturing images using a Raspberry Pi camera  
- Processing and interpreting analog pointer gauges and digital LCD displays  
- Executing lightweight AI models locally (**YOLOv8n, YolactEdge-lite, OCR**)  
- Sending the extracted values to **SCADA-LTS** via HTTP/MQTT  
- Displaying real-time data in **Grafana dashboards**  

The system enables the modernization and digital integration of legacy equipment **without modifying the original instruments**.

---

## Key Features

- **Edge AI** on Raspberry Pi 5 with Hailo-8L accelerator (13 TOPS)  
- Analog gauge reading: pointer detection, angle estimation, scale interpretation  
- Digital LCD reading: OCR via Tesseract  
- OpenCV pre-processing: filtering, segmentation, thresholding  
- Real-time inference (< 0.5 s latency)  
- SCADA integration using open protocols  
- Fully open-source architecture  



