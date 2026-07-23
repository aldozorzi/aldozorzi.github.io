---
title: "Ski Telemetry App"
description: "From Chinese Sensors to Turn Analysis"
date: 2026-04-12
draft: false
---

# Building a Ski Telemetry App

> *From Chinese Sensors to Turn Analysis*

Over the past month and a half, I designed and developed a ski telemetry app. I’m sharing the story here in case anyone is interested in the process. Be warned: this is strictly for ski and software engineering nerds.

A quick disclaimer: this isn't a finished, polished product. It’s a personal project I worked on in my spare time, with all the usual trade-offs that implies: errors, second-guessing, DIY fixes, and a few unexpected wins.

My goal was to find an objective way to measure specific data points, allowing me to analyze the core mechanics of a ski turn rather than just its visual appearance.

---

### Phase 1: Prototyping and Hardware Choices

It all started with a chat conversation with Gemini and a brainstorming session on feasibility. I had never worked with external sensors before, so I had to figure out how to structure the whole architecture. By the end of the conversation, I felt optimistic enough to order sensors from China: two M5StickC Plus2 units based on the ESP32, featuring integrated IMUs. Delivery time: 20 days. 

In the future, I might assemble something custom with a higher-end IMU. However, lacking a 3D printer, making a custom case wasn't an option, so I opted for an off-the-shelf platform. Not ideal from a purely technical standpoint, but practical for getting started.

Instead of waiting idling for delivery, I pulled out an old ESP32 board I had bought years ago for testing and thrown in a drawer. The platform was identical to the incoming sensors, except it lacked accelerometers and gyroscopes. I wrote a firmware to simulate accelerometer and gyroscope data using a sine wave. This allowed me to start working on the BLE connection and data packet structure ahead of time. The data was fake, but it was essential for building the communication layer between the sensor and the mobile app.

---

### Phase 2: App Development and Data Engine

For the app, I picked C# with .NET MAUI so it could run cross-platform on both Android and iOS. I started with the core service modules: BLE pairing, connection management, and settings — boring but necessary infrastructure.

The first major architectural decision was to store raw data exactly as it came off the sensors, without any processing. This turned out to be the most valuable decision of the project. Whenever I realized a formula needed tweaking, I could re-apply it to the exact same recorded data without having to put on ski boots and hit the slopes again. This drastically accelerated subsequent development. 

During this phase, I started building the data analysis engine, designed to process both real-time data during a run and offline logs after the fact. Integrating a text-to-speech engine to read out turn results right after completion turned out to be the easiest part of the whole build.

---

### Phase 3: Zero-Cost Backend Infrastructure

In parallel, I set up the backend. I wanted the ability to share sessions with my coach, which required a database, storage, and hosting. 

The main constraint: zero budget. 

The stack selected: **Vercel**, **NeonDB**, and **Cloudflare** — all running on free tiers. It handles the modest load of a single user just fine. Scaling up can wait for later.

---

### Phase 4: Hardware Integration and Energy Optimization

When the sensors finally arrived, I finalized the firmware to capture real sensor data and set up inter-device communication via ESP-NOW. I implemented a master/slave layout: the slave transmits its readings to the master over ESP-NOW; the master bundles both sets of data into single packets and streams everything to the smartphone over BLE. This way, the phone maintains only one active connection, reducing link instability.

Because the internal batteries are tiny, power optimization was critical. The sensors transmit strictly during an active session. Otherwise, they stay idle without polling the accelerometer or gyroscope, waiting for the next connection trigger from the app.

---

### Phase 5: Field Tests, Mounting, and Sensor Drift

The initial field tests were disappointing but instructive. The stock plastic mounts were too flexible; even tiny vibrations relative to the boot corrupted the data readings. 

To fix this, I dismantled an old PC case, cut out two aluminum mounting plates, and combined them with double-sided tape, neoprene, and two cargo straps resembling a crude booster strap. The result was a rugged, highly stable mounting setup.

Field testing also revealed that the curve-detection algorithms written against simulated sine data failed completely on real-world inputs due to excessive noise. I discarded the accelerometer-based detection model and switched to gyroscope readings, which proved significantly more reliable in this environment.

Calculating G-forces was the most difficult milestone. It required multiple firmware iterations: changing scale ranges, sampling frequencies, and applying different digital filters. It was solved through trial and error rather than pure intuition. 

BLE stability was another headache: phantom disconnects and silent data drops occurred without explicit error logs. The app would fail to reconnect, and tracing the root cause took several iterative debug cycles before reaching acceptable stability.

I also encountered gyroscope drift in the field: even while stationary, error accumulated over time, causing the system to register movement. I implemented an initial calibration step followed by an automatic recalibration routine. This routine triggers automatically when the system detects the user has been stationary long enough — for instance, at the top of a run while awaiting the audio prompt to start.

---

### Outcomes and Next Steps

The most rewarding part of the project was discussing performance metrics with my ski instructor and mentor. He explained how he visually evaluates a turn on the slope, what makes an edge angle efficient, and where kinetic energy is lost. Translating those real-world mechanics into mathematical formulas was the most engaging challenge of the project. 

From those discussions, the **"performance index"** was created — a metric designed to summarize the overall efficiency of every turn. It is still unrefined, but it already yields interesting insights.

The first phase of the project is complete and fully functional. Over the summer, I plan to continue testing using inline skates to evaluate whether the telemetry model applies equally well off the snow.