# High-Resolution Video-Oculograph for Diagnosis and Prevention

## Overview
Oculometry, also known as eye-tracking, is a process used to determine the point of ocular fixation and to measure the movement of the pupil relative to the head. This technology has led to the development of various devices and tools with a wide range of applications, from assistance (e.g., systems that enable paralyzed individuals to communicate) and rehabilitation to estimating attention and concentration.

Modern eye-tracking techniques allow for completely non-invasive detection of movements by using cameras equipped with filters that operate exclusively in the infrared range. This ensures the safety and lack of contraindications for the patient.

Eye-tracking algorithms can be applied either during the video acquisition (real-time) or afterward. The choice between these approaches usually affects factors such as pupil position accuracy and processing time, and it depends on the specific problem to be solved.

## Thesis Focus
This thesis centers on the development of a high spatial resolution video-oculograph capable of diagnosing conditions such as strabismus, and serving as an effective tool for the prevention of amblyopia.

The eye-tracking system used in this work consists of an infrared-sensitive camera, a visible light removal filter, and a computer. The system also includes an algorithm responsible for reconstructing the pupil's trajectory from the captured sequence of frames.

Additionally, a chin rest was employed to ensure the correct and stable positioning of the subject in front of the camera. Infrared illuminators were placed in front of the subject, activated by TTL pulses, though this aspect is not discussed in depth.

## Challenges and Objectives
One challenge that arose during the development of this device was ensuring that the infant's gaze did not move out of the camera's field of view during video acquisition, as this would render the test ineffective. Due to the inability to rely on the subject's cooperation, the potential application of a real-time algorithm was considered. However, even with the most efficient real-time algorithms, there is a trade-off in terms of spatial resolution.

The system operates by capturing frames at a rate of 100 Hz, meaning that the implemented algorithm must be capable of identifying the center of the pupil within a 10ms time window.

The objective of this thesis is to experiment with whether it is possible to implement an algorithm with the described characteristics, not necessarily achieving optimal spatial resolution, but meeting the required time constraints by leveraging neural networks and deep learning techniques.
