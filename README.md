# Hand Eye Coordination Project - ABB IRB120

RME40003 Robot Systems Design, Swinburne University of Technology.

The PC uses OpenCV to locate red, green and blue blocks and convert camera positions into robot coordinates. It sends stack and return plans to an ABB IRB120 with a SCHUNK gripper. In dynamic mode, the robot returns HOME after each transfer while the camera checks the result.

## Files

- `pc/main.cpp`: final PC program, including vision, calibration, planning, dashboard and self-tests.
- `robot/MainModule.mod`: ABB RAPID program for robot motion and command handling.
- `robot/chloec11_2.pgf`: robot program file referencing the RAPID module.
- `pc/*.yml` and reference images: saved settings and backgrounds from the project cell.
- `pc/Robots-Comms-Serial.sln`: Visual Studio solution.
- `pc/libserialport-cmake-master`: bundled third-party serial library source and its original README.
- `pc/HandEye_COM_Direct_v2 (1).cpp`: earlier PC source retained for reference; the Visual Studio project builds `main.cpp`.

## Setup

The PC program uses C++, OpenCV 4 and libserialport on Windows. The supplied project contains local build settings and may need its include paths, library paths and toolset adjusted on another computer. `OpenCV_Debug.props` references OpenCV 4.6.0 debug libraries through `OPENCV_DIR`. The serial library README explains its CMake build.

Open the solution, select the appropriate x64 configuration, build the dependencies and program, then run the resulting executable with the serial port, for example:

```text
Robots-Comms-Serial.exe COM3
```

Run from the `pc` folder so the relative settings and image paths resolve. Load the robot program on the controller and check the cell I/O mapping for `dg`, `doGripOpen` and `doGripClose`.

The saved calibration, taught targets, workspace limits and offsets belong to the original hardware setup. Check them against the actual cell before running motion. Capture new backgrounds and calibrate again if the camera or workspace changes.

## Modes

- Static: pick from the nine taught positions P1-P9.
- Dynamic: use camera coordinates to stack blocks and return them to free positions.
- Calibrate: set the camera-to-robot mapping and capture lighting references.
- Test: check individual picks and adjust offsets.

## Checks without robot motion

The source includes self-test entry points such as:

```text
Robots-Comms-Serial.exe --self-test
Robots-Comms-Serial.exe --v10-self-test
Robots-Comms-Serial.exe --calibration-self-test
Robots-Comms-Serial.exe --ui-audit-self-test
```

See `main()` for the full list and any required arguments. Uploading this repository does not validate the software on another robot cell.

## Communication and cell safety

Communication is one-way from PC to robot, with no robot acknowledgement. Values are held for 250 ms and separated by zero. The source headers describe the frame format and commands.

Software workspace checks and camera supervision are task checks. Controller STOP and cell guarding provide the safety measures. Gripper opening and closing are timed; the robot has no grip feedback signal.

## Repository contents

This is the final project snapshot from `1_Final`. Build outputs, Visual Studio caches, duplicate ZIP archives, session captures, diagnostic logs and the previous runtime cycle journal are excluded. The original local files are retained separately.

The bundled libserialport code is third-party work; see its README and source notices. No new license is assigned to it by this repository.
