# MLGestureControl
To build an ML model on top of MediaPipe, one needs to collect labeled hand motion sequences, extract normalized temporal features from hand landmarks, train a lightweight intent classifier, and integrate it with a rule-based robot controller.

Pi (HamBot)                             Mac
┌──────────────────┐                   ┌──────────────────────┐
│ picamera2 capture│─Channel 1 Video──▶│ MediaPipe 제스처 인식  │
│                  │                   │                      │
│ motor control    │◀─Channel 2 command│ OPEN/CLOSE/POINTER/OK │
└──────────────────┘                   └──────────────────────┘
port 5000 (video)                   port 5001 (cmd)