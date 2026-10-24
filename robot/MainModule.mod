MODULE MainModule

    ! Hand eye coordination - ABB IRB120 and SCHUNK gripper (RME40003)
    ! main.cpp handles the camera and planning. This module reads commands from dg and moves the robot.
    ! Build bottom first, return top first. Go HOME after each dynamic transfer for the camera check.
    !
    ! Serial format (PC to robot, no reply)
    !   Every value is held on dg for 250 ms with a 0 between values:
    !     0, START, 0, field, 0, field, ... 0, END, 0
    !   Checksum = (sum of fields) MOD 251 + 1.   XY in 0.5 mm steps:
    !     X = 100 + q/2,  Y = -400 + q/2,  q = (hi - 1) * 128 + (lo - 1)
    !   226..229  mode select: 1 static, 2 dynamic, 3 test, 4 calibrate (code - 225)
    !   220..221  dynamic stack plan : token, pattern, 3 x XY, checksum
    !   222..223  dynamic return plan: token, 3 x XY, checksum
    !   234..235  control: token, action (1 HOLD, 2 HOME, 3 RESET, 4 PERMIT), step, checksum
    !   236..237  camera target: token, mode (1 re-pick, 2 new next target), step,
    !             attempt, XY, checksum
    !   232..233  TEST page: token, action, XY, checksum   (230..231 older 0.1 mm format)
    !   240..241  static stack plan: site 161..163, L1 177..185, L2 193..201, L3 209..217
    !   250..251  static return plan: 128+site, L3 16+P, L2 32+P, L1 48+P
    !   The token identifies one cycle: stale or repeated frames are refused.
    !
    ! Cell safety
    !   The checks in this module (XY limits, tokens, checkpoints) protect the task, not people.
    !   The controller STOP and the cell guarding are the safety measures. The gripper has no
    !   feedback signal: a grip is never confirmed by the robot, only by the PC camera.
    !
    ! Where to find things
    !   Data:  A taught targets, B settings, C limits, D dynamic cycle state,
    !          E test page state, F static mode state, G diagnostics
    !   1 start-up and main loop      2 serial receiver        3 dynamic cycle
    !   4 TEST page                   5 static mode            6 targets, motion and gripper
    ! ------------------------------------------------------------

    ! ------------------------------------------------------------
    ! A  TAUGHT TARGETS
    !   P1..P9: nine pickup positions (70 x 75 mm grid). Above = 286.03, grip = 200.81 mm.
    !   S1..S3: stack sites at Y 185.98. Levels 1..3 at Z 200.81 / 236.81 / 272.81 (36 mm pitch).
    !   Pickup and stack targets share one tool orientation. Camera XY targets copy P7, change X/Y only.
    !   Returned blocks use the P7 orientation with the return heights returnDropZ / returnAboveZ (B).
    ! ------------------------------------------------------------
    CONST robtarget pHome := [[276.18,10.39,366.51],[3.1863E-05,0.0555729,-0.998455,4.38459E-05],[0,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP1Above := [[372.65,-82.80,286.03],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP1Pick := [[372.65,-82.80,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP2Above := [[372.65,-7.80,286.03],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP2Pick := [[372.65,-7.80,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP3Above := [[372.65,67.20,286.03],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP3Pick := [[372.65,67.20,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP4Above := [[442.65,-82.80,286.03],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP4Pick := [[442.65,-82.80,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP5Above := [[442.65,-7.80,286.03],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP5Pick := [[442.65,-7.80,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP6Above := [[442.65,67.20,286.03],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP6Pick := [[442.65,67.20,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP7Above := [[512.65,-82.80,286.03],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP7Pick := [[512.65,-82.80,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP8Above := [[512.65,-7.80,286.03],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP8Pick := [[512.65,-7.80,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[-1,0,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP9Above := [[512.65,67.20,286.03],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pP9Pick := [[512.65,67.20,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS1Above := [[372.65,185.98,338.91],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS1Level1 := [[372.65,185.98,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS1Level2 := [[372.65,185.98,236.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS1Level3 := [[372.65,185.98,272.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS2Above := [[442.65,185.98,338.91],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS2Level1 := [[442.65,185.98,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS2Level2 := [[442.65,185.98,236.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS2Level3 := [[442.65,185.98,272.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS3Above := [[512.65,185.98,338.91],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS3Level1 := [[512.65,185.98,200.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS3Level2 := [[512.65,185.98,236.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget pS3Level3 := [[512.65,185.98,272.81],[3.44761E-05,0.0546444,-0.998506,-9.54626E-06],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];


    ! ------------------------------------------------------------
    ! B  SETTINGS (change on the pendant: Program Data)
    ! ------------------------------------------------------------
    ! FALSE = monitor only: every command is checked and reported but nothing moves.
    PERS bool enableMotion := TRUE;
    ! Separate switches for the TEST page and the automatic dynamic cycle.
    PERS bool enableDynamicTest := TRUE;
    PERS bool enableDynamicCycle := TRUE;
    ! Physical stack site used for every stack: 1, 2 or 3 (S1/S2/S3 on the dashboard is the colour pattern).
    PERS num stackSite := 1;
    ! Mode last sent by the dashboard (set by the PC, not a setting): 0 none, 1 static, 2 dynamic, 3 test, 4 calibration.
    PERS num uiMode := 0;
    ! TRUE prints every received byte and joint values (serial or configuration fault finding).
    PERS bool verboseDiagnostics := FALSE;
    ! Heights for a block RETURNED from the stack to the table (tool0 flange Z, mm). Dynamic only.
    !   returnDropZ : the gripper opens here. Start = taught table grip height (pP7Pick 200.81).
    !                 Raise it 1..3 mm to release just above the table instead of on it.
    !   returnAboveZ: approach and retract height above the drop point (pP7Above 286.03).
    ! Pickups from the table and the stack levels keep their taught heights. See ReturnTargets.
    PERS num returnDropZ := 205.81;
    PERS num returnAboveZ := 286.03;

    ! ------------------------------------------------------------
    ! C  LIMITS AND CLEARANCES
    ! ------------------------------------------------------------
    ! Minimum centre spacing between two pickups (open fingers 22.8 x 71 mm, 37 mm block at 12 deg,
    ! 3 mm margin). Must match the PC values 47 / 61 mm.
    CONST num v10GapX := 47;
    CONST num v10GapY := 61;
    ! Inner TEST region for the older 230 frame.
    CONST num dynamicMinX := 352.7;
    CONST num dynamicMaxX := 532.6;
    CONST num dynamicMinY := -102.8;
    CONST num dynamicMaxY := 87.2;
    ! Reachable area for every camera XY: X >= 200 mm and 360..540 mm from the base (see V10PointOK).
    CONST num dynamicWideMinX := 200;
    CONST num dynamicReachMin := 360;
    CONST num dynamicReachMax := 540;
    ! Start-up lifts to this height before moving HOME (highest taught approach).
    CONST num startupSafeZ := 338.91;
    ! Allowed returnDropZ: 2 mm below to 20 mm above the taught table grip height (pP7Pick).
    ! returnAboveZ must be at least 40 mm above returnDropZ and not above startupSafeZ.
    ! Outside these limits ReturnTargets uses the taught P7 heights and says so on the pendant.
    CONST num returnDropMinDZ := -2;
    CONST num returnDropMaxDZ := 20;
    CONST num returnAboveMinGap := 40;

    ! ------------------------------------------------------------
    ! D  DYNAMIC CYCLE STATE
    !   PERS values survive a program stop, so an unfinished cycle is still known after a restart.
    ! ------------------------------------------------------------
    ! Phase: 0 idle, 1 stacking, 2 stack built, 3 returning, 4 complete, 5 recovery hold.
    PERS num v10Phase := 0;
    ! Token of the active cycle, chosen pattern and the stack site fixed at the start of the cycle.
    PERS num v10Token := 0;
    PERS num v10Pattern := 0;
    PERS num v10Site := 1;
    ! Current pickup and drop XY per stack level (a camera retarget overwrites them).
    PERS num v10PickX{3} := [0,0,0];
    PERS num v10PickY{3} := [0,0,0];
    PERS num v10DropX{3} := [0,0,0];
    PERS num v10DropY{3} := [0,0,0];
    ! Last camera re-pick / retarget attempt executed. A repeated attempt never moves.
    PERS num v13LastAttempt := 0;
    ! Checkpoint flags: robot waiting at HOME, the step it waits at, permit received, cycle aborted,
    ! recovery hold latched, and whether HOME was already reached at this checkpoint.
    VAR bool v11AtCheckpoint := FALSE;
    VAR num v11ExpectedStep := 0;
    VAR bool v11Permit := FALSE;
    VAR bool v11AbortCycle := FALSE;
    VAR bool v11FaultLatch := FALSE;
    VAR bool v11HomeDone := FALSE;

    ! ------------------------------------------------------------
    ! E  TEST PAGE STATE
    ! ------------------------------------------------------------
    ! Stage: 0 idle, 1 hovering above the block, 2 block lifted, -1 moving.
    PERS num dynamicStage := 0;
    ! Token and XY locked at the HOVER command; PICK and REPLACE must repeat the same bytes.
    PERS num dynamicToken := 0;
    PERS num dynamicX := 0;
    PERS num dynamicY := 0;
    PERS num dynamicWireXY{4} := [0,0,0,0];
    PERS num dynamicWireMode := 1;
    VAR robtarget dynamicAbove;
    VAR robtarget dynamicGrip;
    ! TEST stack build: blocks picked one by one and stacked at S1, later put back where they came from.
    VAR num buildCount := 0;
    VAR num buildX{3};
    VAR num buildY{3};

    ! ------------------------------------------------------------
    ! F  STATIC MODE STATE
    ! ------------------------------------------------------------
    ! Raw received bytes (shown on the pendant when a frame is rejected).
    VAR num rawStack := 0;
    VAR num rawL1 := 0;
    VAR num rawL2 := 0;
    VAR num rawL3 := 0;
    VAR num rawRS := 0;
    VAR num rawR1 := 0;
    VAR num rawR2 := 0;
    VAR num rawR3 := 0;
    ! Decoded plan: stack site, pickup P for L1..L3, return P for L1..L3.
    VAR num stackNo := 0;
    VAR num pick1 := 0;
    VAR num pick2 := 0;
    VAR num pick3 := 0;
    VAR num ret1 := 0;
    VAR num ret2 := 0;
    VAR num ret3 := 0;
    VAR bool planValid := FALSE;
    ! Stored plans per stack site and the order they arrived in.
    VAR num storedPick{3,3};
    VAR num returnP{3,3};
    VAR num stackOrder{3};
    VAR num planCount := 0;

    ! ------------------------------------------------------------
    ! G  DIAGNOSTICS
    !   Last motion step and joint values, readable in Program Data after a fault. Never used to move.
    ! ------------------------------------------------------------
    VAR string lastMotionStep := "NOT STARTED";
    VAR jointtarget lastMotionJoints;

    ! ------------------------------------------------------------
    ! SECTION 1  START-UP AND MAIN LOOP
    ! ------------------------------------------------------------

    ! Start-up: report the saved state, run the start-up check, then listen on dg forever.
    ! Each command starts with a START value; the matching routine reads and checks the rest.
    PROC main()
        VAR bool ok := FALSE;
        VAR num startByte := 0;
        VAR num endByte := 0;
        ! Mixed arm configurations between taught targets: let the controller pick the nearest.
        ! Better long-term fix: re-teach pS3Above and P3/P6/P9 with matching confdata.
        ConfJ \Off;
        ConfL \Off;
        ! Test token/XY/stage persist together. Do not erase after a stop. Inspect physical state first.
        v11AtCheckpoint:=FALSE;
        TPWrite "HAND EYE V13.8 started";
        ReportSession;
        StartupCheck;
        TPWrite "Listening to the PC on dg";
        TPWrite "  dg now" \Num:=GInput(dg);
        WHILE TRUE DO
            planValid := FALSE;
            rawStack := 0;
            rawL1 := 0;
            rawL2 := 0;
            rawL3 := 0;
            startByte := 0;
            endByte := 0;
            TPWrite "READY: waiting for the PC";
            WaitCommand startByte, ok;
            IF ok AND startByte>=226 AND startByte<=229 THEN
                SelectDashboard startByte;
            ELSEIF ok AND startByte = 234 THEN
                ReceiveV11Control;
            ELSEIF ok AND startByte = 236 THEN
                ! Re-pick/retarget is valid only while waiting at a checkpoint: read and refuse.
                ReceiveV13Target 0;
            ELSEIF ok AND startByte = 220 THEN
                TPWrite "RX: dynamic stack plan, checking it";
                ReceiveV10 1;
            ELSEIF ok AND startByte = 222 THEN
                TPWrite "RX: return plan, checking it";
                ReceiveV10 2;
            ELSEIF ok AND startByte = 230 THEN
                ReceiveDynamicTest 0;
            ELSEIF ok AND startByte = 232 THEN
                TPWrite "RX: TEST page command";
                ReceiveDynamicTest 1;
            ELSEIF ok AND startByte = 240 THEN
                TPWrite "RX: static stack plan, checking it";
                CaptureByte rawStack, ok;
                IF ok THEN
                    CaptureByte rawL1, ok;
                ENDIF
                IF ok THEN
                    CaptureByte rawL2, ok;
                ENDIF
                IF ok THEN
                    CaptureByte rawL3, ok;
                ENDIF
                IF ok THEN
                    CaptureByte endByte, ok;
                ENDIF
                IF ok AND endByte = 241 THEN
                    planValid := DecodePlan();
                ENDIF
                IF planValid THEN
                    TPWrite "PC -> STATIC PLAN to S"+NumToStr(stackNo,0)+": P"+NumToStr(pick1,0)+", P"+NumToStr(pick2,0)+", P"+NumToStr(pick3,0);
                    TPWrite "  (bottom, middle, top)";
                    ! Ensure sender has finished and dg returned to idle.
                    AwaitValue 0, ok;
                    IF (ok = FALSE) THEN
                        planValid := FALSE;
                        TPWrite "NO IDLE - PLAN REJECTED";
                    ENDIF
                    IF planValid THEN
                        IF uiMode<>0 AND uiMode<>1 THEN
                            TPWrite "STATIC REJECTED: SELECT STATIC MODE";
                            ReportSession;
                        ELSEIF dynamicStage <> 0 OR v10Phase = 1 OR v10Phase = 2 OR v10Phase = 3 THEN
                            TPWrite "STATIC BLOCKED: DYNAMIC UNFINISHED";
                            ReportSession;
                            TPWrite "TEST STAGE" \Num:=dynamicStage;
                            TPWrite "AUTO PHASE" \Num:=v10Phase;
                        ELSEIF enableMotion THEN
                            IF NOT StackAlreadyStored(stackNo) THEN
                                IF SourceAvailable() THEN
                                    StorePlan;
                                    TPWrite "PLAN STORED; TOTAL PLANS" \Num:=planCount;
                                    TPWrite "NOW RUNNING S" \Num:=stackNo;
                                    BuildOneStack stackNo;
                                    TPWrite "STACK FINISHED S" \Num:=stackNo;
                                    HandleReturn stackNo;
                                    TPWrite "WAIT FOR NEXT PLAN";
                                ELSE
                                    TPWrite "DUPLICATE PICKUP POSITION - PLAN REJECTED";
                                ENDIF
                            ELSE
                                TPWrite "STACK STILL STORED - RETURN MUST FINISH";
                                TPWrite "STORED PLANS" \Num:=planCount;
                            ENDIF
                        ELSE
                            TPWrite "MOTION LOCKED - MONITOR ONLY";
                        ENDIF
                    ENDIF
                ELSE
                    TPWrite "BAD FRAME / INVALID PLAN";
                    TPWrite "Raw stack" \Num:=rawStack;
                    TPWrite "Raw L1" \Num:=rawL1;
                    TPWrite "Raw L2" \Num:=rawL2;
                    TPWrite "Raw L3" \Num:=rawL3;
                    TPWrite "Raw END" \Num:=endByte;
                ENDIF
            ELSEIF ok THEN
                TPWrite "IGNORED: unknown start value" \Num:=startByte;
                ReportSession;
            ENDIF
            AwaitValue 0, ok;
        ENDWHILE
    ENDPROC

    ! Pendant text helpers: "(412.5, -96.0)" and the level names used in messages.
    FUNC string XYText(num x, num y)
        RETURN "("+NumToStr(x,1)+", "+NumToStr(y,1)+")";
    ENDFUNC

    FUNC string LevelName(num level)
        VAR string name := "level ?";
        TEST level
        CASE 1: name := "L1 bottom";
        CASE 2: name := "L2 middle";
        CASE 3: name := "L3 top";
        ENDTEST
        RETURN name;
    ENDFUNC

    ! Restored start-up routine: lift straight up, go HOME, open the gripper.
    ! Idle: automatic. Unfinished cycle: a block may be held, so the operator chooses on the pendant.
    ! No cycle state changes here; an unfinished dynamic cycle still needs the normal PC RESET.
    PROC StartupCheck()
        VAR num answer := 1;
        VAR bool idle;
        IF enableMotion = FALSE THEN
            TPWrite "STARTUP: enableMotion FALSE - no startup motion";
            RETURN;
        ENDIF
        idle := (v10Phase = 0 OR v10Phase = 4) AND dynamicStage = 0;
        IF idle = FALSE THEN
            TPWrite "STARTUP: previous cycle unfinished; a block may be held";
            TPReadFK answer, "Startup: lift, HOME and open gripper?", "HOME+OPEN", "HOME ONLY", stEmpty, stEmpty, "SKIP";
            IF answer = 5 THEN
                TPWrite "STARTUP: skipped by operator";
                RETURN;
            ENDIF
        ENDIF
        StartupLiftHome;
        IF answer = 1 THEN
            TPWrite "STARTUP: opening gripper at HOME";
            OpenGripper;
        ELSE
            TPWrite "STARTUP: gripper left as it is";
        ENDIF
        TPWrite "STARTUP CHECK COMPLETE: at HOME";
    ENDPROC

    ! Leave a low pose vertically first, so the joint move HOME cannot swing the fingers through blocks.
    PROC StartupLiftHome()
        VAR robtarget here;
        here := CRobT(\Tool:=tool0 \WObj:=wobj0);
        IF here.trans.z < startupSafeZ THEN
            TPWrite "STARTUP: lifting vertically to safe height";
            here.trans.z := startupSafeZ;
            MoveL here, v100, fine, tool0;
        ENDIF
        TPWrite "STARTUP: moving HOME";
        MoveJ pHome, v100, fine, tool0;
    ENDPROC

    ! Mode frame 226..229 from the dashboard tabs. Refused while a session is unfinished. No motion.
    PROC SelectDashboard(num code)
        VAR bool idle;
        VAR num requested;
        requested:=code-225;
        AwaitValue 0,idle;
        IF (idle = FALSE) THEN
            TPWrite "MODE REJECTED: dg NOT IDLE";
            RETURN;
        ENDIF
        IF verboseDiagnostics THEN
            TPWrite "Mode wire code" \Num:=code;
        ENDIF
        IF requested<1 OR requested>4 THEN
            TPWrite "MODE REJECTED: INVALID VALUE";
            RETURN;
        ENDIF
        TEST requested
        CASE 1: TPWrite "PC -> MODE STATIC";
        CASE 2: TPWrite "PC -> MODE DYNAMIC";
        CASE 3: TPWrite "PC -> MODE TEST";
        CASE 4: TPWrite "PC -> MODE CALIBRATE";
        ENDTEST
        IF uiMode=requested THEN
            TPWrite "MODE ALREADY SELECTED - NO RESET";
            ReportSession;
            RETURN;
        ENDIF
        IF planCount<>0 OR dynamicStage<>0 OR buildCount<>0 OR (v10Phase<>0 AND v10Phase<>4) THEN
            TPWrite "MODE BLOCKED: UNFINISHED SESSION";
            TPWrite "FINISH RETURN OR RECOVER ON PENDANT";
            ReportSession;
            RETURN;
        ENDIF
        uiMode:=requested;
        !Phase4 is set only AFTER the final fine HOME move. Keep token history.
        IF v10Phase=4 THEN
            v10Phase:=0;
        ENDIF
        TPWrite "MODE ACCEPTED - NO MOTION";
        IF uiMode=1 THEN
            TPWrite "Waiting for a static stack plan from PC";
        ELSEIF uiMode=2 THEN
            TPWrite "Waiting for a dynamic stack plan from PC";
        ELSEIF uiMode=3 THEN
            TPWrite "Waiting for test HOVER at camera pickup position";
        ELSE
            TPWrite "CALIBRATION: PC MOTION DISABLED";
        ENDIF
        ReportSession;
    ENDPROC

    ! Prints the selected mode and the saved cycle state (used at start-up and after refusals).
    PROC ReportSession()
        TEST uiMode
        CASE 1: TPWrite "Selected mode: STATIC stacking";
        CASE 2: TPWrite "Selected mode: DYNAMIC vision stacking";
        CASE 3: TPWrite "Selected mode: supervised PICK TEST";
        CASE 4: TPWrite "Selected mode: CAMERA CALIBRATION";
        DEFAULT: TPWrite "No dashboard mode selected yet";
        ENDTEST
        TEST v10Phase
        CASE 0: TPWrite "Dynamic cycle: IDLE";
        CASE 1: TPWrite "Dynamic cycle: STACKING / vision checkpoint";
        CASE 2: TPWrite "Stack built; waiting for PC return positions";
        CASE 3: TPWrite "Dynamic cycle: RETURNING blocks top first";
        CASE 4: TPWrite "Dynamic cycle: COMPLETE; robot at HOME";
        CASE 5: TPWrite "RECOVERY HOLD: inspect blocks and pendant";
        DEFAULT: TPWrite "Unknown saved cycle state; inspect before reset";
        ENDTEST
        IF dynamicStage=1 THEN
            TPWrite "Test: at pickup approach; waiting PICK or CANCEL";
        ELSEIF dynamicStage=2 THEN
            TPWrite "Test: block held; waiting REPLACE command";
        ELSEIF dynamicStage<>0 THEN
            TPWrite "Test motion unfinished; recover on pendant";
        ENDIF
        IF planCount>0 THEN
            TPWrite "Static stacks still awaiting return" \Num:=planCount;
        ENDIF
        IF verboseDiagnostics THEN
            TPWrite "Cycle identifier (not an error)" \Num:=v10Token;
        ENDIF
    ENDPROC

    ! ------------------------------------------------------------
    ! SECTION 2  SERIAL RECEIVER
    ! ------------------------------------------------------------

    ! Waits up to about 12 s until dg shows `target` steadily (read twice, 40 ms apart).
    PROC AwaitValue(num target, VAR bool ok)
        VAR num i;
        ok := FALSE;
        FOR i FROM 1 TO 600 DO
            IF GInput(dg) = target THEN
                WaitTime 0.04;
                IF GInput(dg) = target THEN
                    ok := TRUE;
                    RETURN;
                ENDIF
            ENDIF
            WaitTime 0.02;
        ENDFOR
        TPWrite "RX TIMEOUT WAITING FOR VALUE" \Num:=target;
        TPWrite "CURRENT dg" \Num:=GInput(dg);
    ENDPROC

    ! Reads one field: waits for the 0 separator, then a steady non zero value.
    PROC CaptureByte(VAR num value, VAR bool ok)
        VAR num i;
        value := 0;
        AwaitValue 0, ok;
        IF (ok = FALSE) THEN
            RETURN;
        ENDIF
        ok := FALSE;
        FOR i FROM 1 TO 600 DO
            value := GInput(dg);
            IF value <> 0 THEN
                WaitTime 0.04;
                IF GInput(dg) = value THEN
                    ok := TRUE;
                    IF verboseDiagnostics THEN
                        TPWrite "Wire byte (diagnostic)" \Num:=value;
                    ENDIF
                    RETURN;
                ENDIF
            ENDIF
            WaitTime 0.02;
        ENDFOR
        TPWrite "Incomplete packet: next field did not arrive";
        TPWrite "CURRENT dg" \Num:=GInput(dg);
    ENDPROC

    ! Idle listening for the next START value (no timeout).
    PROC WaitCommand(VAR num value, VAR bool ok)
        VAR num observed;
        AwaitValue 0,ok;
        IF (ok = FALSE) THEN
            RETURN;
        ENDIF
        WHILE TRUE DO
            observed:=GInput(dg);
            IF observed<>0 THEN
                WaitTime 0.04;
                IF GInput(dg)=observed THEN
                    value:=observed;
                    ok:=TRUE;
                    RETURN;
                ENDIF
            ENDIF
            WaitTime 0.02;
        ENDWHILE
    ENDPROC

    ! Reads past the rest of a frame up to its END value, so its fields are not taken as new commands.
    PROC SkipFrame(num endByte)
        VAR num skipped;
        VAR num k;
        VAR bool good;
        FOR k FROM 1 TO 16 DO
            CaptureByte skipped,good;
            IF (good = FALSE) OR skipped=endByte THEN
                RETURN;
            ENDIF
        ENDFOR
    ENDPROC

    ! ------------------------------------------------------------
    ! SECTION 3  DYNAMIC CYCLE (camera positions, HOME checkpoints)
    ! ------------------------------------------------------------

    ! Dynamic stack plan (mode 1, frame 220) or return plan (mode 2, frame 222).
    ! All three XY are checked (checksum, reach, spacing, mode, token) before anything moves.
    ! Stacking: pick L1, L2, L3 and stop at a HOME checkpoint after each one.
    ! Returning: take L3, L2, L1 off the stack, again with a checkpoint after each block.
    PROC ReceiveV10(num mode)
        VAR num b{15};
        VAR num n;
        VAR num i;
        VAR num j;
        VAR num offset;
        VAR num finish;
        VAR num checksum := 0;
        VAR num px{3};
        VAR num py{3};
        VAR bool ok;
        VAR robtarget pAbove;
        VAR robtarget pGrip;
        VAR robtarget sAbove;
        VAR robtarget sGrip;
        checksum := 0;
        n := 15;
        IF mode = 2 THEN
            n := 14;
        ENDIF
        FOR i FROM 1 TO n DO
            CaptureByte b{i},ok;
            IF ok = FALSE THEN
                TPWrite "PLAN REJECTED: frame incomplete (no move)";
                RETURN;
            ENDIF
        ENDFOR
        CaptureByte finish,ok;
        IF ok = FALSE OR finish <> 219+mode*2 THEN
            TPWrite "PLAN REJECTED: wrong end value (no move)";
            RETURN;
        ENDIF
        AwaitValue 0,ok;
        IF ok = FALSE THEN
            RETURN;
        ENDIF
        FOR i FROM 1 TO n-1 DO
            checksum := checksum+b{i};
        ENDFOR
        IF b{n} <> (checksum MOD 251)+1 OR b{1}<1 OR b{1}>127 THEN
            TPWrite "PLAN REJECTED: checksum or token (no move)";
            RETURN;
        ENDIF
        offset := 2;
        IF mode = 2 THEN
            offset := 1;
        ELSEIF b{2}<1 OR b{2}>3 THEN
            RETURN;
        ENDIF
        ! Show what the PC sent, in plain words, before the remaining checks.
        IF mode=1 THEN
            TPWrite "PC -> STACK PLAN (cycle "+NumToStr(b{1},0)+", pattern "+NumToStr(b{2},0)+")";
        ELSE
            TPWrite "PC -> RETURN PLAN (cycle "+NumToStr(b{1},0)+")";
        ENDIF
        FOR i FROM 1 TO 3 DO
            j := offset+(i-1)*4;
            px{i}:=100+((b{j+1}-1)*128+b{j+2}-1)/2;
            py{i}:=-400+((b{j+3}-1)*128+b{j+4}-1)/2;
            IF mode=1 THEN
                TPWrite "  "+LevelName(i)+": pick at "+XYText(px{i},py{i});
            ELSE
                TPWrite "  "+LevelName(i)+": drop at "+XYText(px{i},py{i});
            ENDIF
        ENDFOR
        FOR i FROM 1 TO 3 DO
            j := offset+(i-1)*4;
            IF b{j+1}>16 OR b{j+2}>128 OR b{j+3}>16 OR b{j+4}>128 THEN
                RETURN;
            ENDIF
            px{i}:=100+((b{j+1}-1)*128+b{j+2}-1)/2;
            py{i}:=-400+((b{j+3}-1)*128+b{j+4}-1)/2;
            IF V10PointOK(px{i},py{i}) = FALSE THEN
                TPWrite "PLAN REJECTED: "+LevelName(i)+" "+XYText(px{i},py{i})+" not reachable";
                RETURN;
            ENDIF
        ENDFOR
        FOR i FROM 1 TO 2 DO
            FOR j FROM i+1 TO 3 DO
                IF Abs(px{i}-px{j})<v10GapX AND Abs(py{i}-py{j})<v10GapY THEN
                    TPWrite "PLAN REJECTED: blocks too close (no move)";
                    RETURN;
                ENDIF
            ENDFOR
        ENDFOR
        IF enableMotion = FALSE THEN
            TPWrite "PLAN REFUSED: enableMotion is FALSE";
            ReportSession;
            RETURN;
        ENDIF
        IF uiMode<>0 AND uiMode<>2 THEN
            TPWrite "PLAN REFUSED: select DYNAMIC on the PC";
            ReportSession;
            RETURN;
        ENDIF
        IF verboseDiagnostics THEN
            TPWrite "Cycle identifier" \Num:=b{1};
        ENDIF
        IF mode=1 THEN
            TPWrite "PLAN OK: stacking bottom to top";
        ELSE
            TPWrite "PLAN OK: returning top block first";
        ENDIF
        IF mode=1 THEN
            TEST b{2}
            CASE 1: TPWrite "Pattern: bottom BLUE, middle RED, top GREEN";
            CASE 2: TPWrite "Pattern: bottom BLUE, middle GREEN, top RED";
            CASE 3: TPWrite "Pattern: bottom RED, middle GREEN, top BLUE";
            ENDTEST
        ENDIF
        IF enableDynamicCycle = FALSE THEN
            TPWrite "PLAN REFUSED: enableDynamicCycle is FALSE";
            RETURN;
        ENDIF
        IF dynamicStage <> 0 OR buildCount <> 0 OR planCount <> 0 THEN
            TPWrite "PLAN REFUSED: static or test session open";
            ReportSession;
            RETURN;
        ENDIF
        IF mode = 1 THEN
            IF v10Phase <> 0 AND v10Phase <> 4 THEN
                TPWrite "PLAN REFUSED: last cycle not finished";
                RETURN;
            ENDIF
            IF b{1}=v10Token THEN
                TPWrite "PLAN IGNORED: same cycle sent again";
                RETURN;
            ENDIF
            v10Token:=b{1};
            v10Pattern:=b{2};
            IF stackSite<1 OR stackSite>3 THEN
                TPWrite "INVALID STACK SITE - NO MOVE";
                RETURN;
            ENDIF
            v10Site:=stackSite;
            FOR i FROM 1 TO 3 DO
                v10PickX{i}:=px{i};
                v10PickY{i}:=py{i};
            ENDFOR
            ! Persist in-progress BEFORE commanding any motion. Do not reset after faults.
            v11AbortCycle:=FALSE;
            v11FaultLatch:=FALSE;
            v11AtCheckpoint:=FALSE;
            v13LastAttempt:=0;
            v10Phase:=1;
            TPWrite "CYCLE START: stacking at site S" \Num:=v10Site;
            FOR i FROM 1 TO 3 DO
                pAbove:=pP7Above;
                pGrip:=pP7Pick;
                ! V13: read the stored target so a camera retarget at the previous checkpoint applies.
                pAbove.trans.x:=v10PickX{i};pAbove.trans.y:=v10PickY{i};
                pGrip.trans.x:=v10PickX{i};pGrip.trans.y:=v10PickY{i};
                GetStackTargets stackSite,i,sAbove,sGrip;
                TPWrite LevelName(i)+": "+XYText(v10PickX{i},v10PickY{i})+" -> stack S"+NumToStr(v10Site,0);
                V10Transfer pAbove,pGrip,sAbove,sGrip;
                V11Checkpoint i;
                IF v11AbortCycle THEN
                    RETURN;
                ENDIF
            ENDFOR
            MoveJ pHome,v200,fine,tool0;
            v10Phase:=2;
            v11AtCheckpoint:=TRUE;
            v11HomeDone:=TRUE;
            v11ExpectedStep:=0;
            TPWrite "CYCLE: stack built, at HOME";
            TPWrite "CYCLE: waiting for the return plan";
        ELSE
            IF v10Phase<>2 OR b{1}<>v10Token THEN
                ReportSession;
                TPWrite "RETURN REFUSED: no stack or wrong cycle";
                RETURN;
            ENDIF
            FOR i FROM 1 TO 3 DO
                v10DropX{i}:=px{i};v10DropY{i}:=py{i};
            ENDFOR
            v10Phase:=3;
            TPWrite "RETURN START: top block first";
            FOR i FROM 3 TO 1 STEP -1 DO
                GetStackTargets stackSite,i,sAbove,sGrip;
                ReturnTargets v10DropX{i},v10DropY{i},pAbove,pGrip;
                TPWrite LevelName(i)+": stack S"+NumToStr(v10Site,0)+" -> "+XYText(v10DropX{i},v10DropY{i});
                V10Transfer sAbove,sGrip,pAbove,pGrip;
                V11Checkpoint 7-i;
                IF v11AbortCycle THEN
                    RETURN;
                ENDIF
            ENDFOR
            MoveJ pHome,v200,fine,tool0;
            v10Phase:=4;
            v11AtCheckpoint:=TRUE;
            v11HomeDone:=TRUE;
            v11ExpectedStep:=0;
            TPWrite "CYCLE COMPLETE: all blocks returned";
            TPWrite "CYCLE: ready for the next one";
        ENDIF
    ENDPROC

    ! TRUE when XY is reachable and outside the three stack site zones.
    FUNC bool V10PointOK(num x, num y)
        VAR num r;
        VAR num i;
        VAR num sx;
        r := Sqrt(x*x+y*y);
        IF x < dynamicWideMinX OR r < dynamicReachMin OR r > dynamicReachMax THEN
            RETURN FALSE;
        ENDIF
        FOR i FROM 1 TO 3 DO
            sx := 302.65 + i*70;
            IF Abs(x-sx) < 60 AND Abs(y-185.98) < 80 THEN
                RETURN FALSE;
            ENDIF
        ENDFOR
        RETURN TRUE;
    ENDFUNC

    ! One pick and place: approach, open, descend, close, lift, move, descend, open, retract.
    PROC V10Transfer(robtarget fromAbove,robtarget fromGrip,robtarget toAbove,robtarget toGrip)
        TPWrite "PICK: moving to source approach";
        MoveJ fromAbove,v200,fine,tool0;
        OpenGripper;
        TPWrite "PICK: descending to taught grip height";
        MoveL fromGrip,v150,fine,tool0;
        CloseGripper;
        TPWrite "PICK: lifting block clear of source";
        MoveL fromAbove,v150,fine,tool0;
        TPWrite "PLACE: moving to destination approach";
        MoveJ toAbove,v200,fine,tool0;
        TPWrite "PLACE: descending to destination level";
        MoveL toGrip,v150,fine,tool0;
        OpenGripper;
        TPWrite "PLACE: released block; retracting vertically";
        MoveL toAbove,v150,fine,tool0;
    ENDPROC

    ! After each transfer: go HOME (clear of the camera view) and wait for the PC.
    ! Leaves on a permit (234 action 4) or a new next target (236 mode 2). A camera re-pick
    ! (236 mode 1) repeats the transfer and waits here again. A RESET aborts the cycle.
    PROC V11Checkpoint(num checkpointStep)
        VAR num sb;
        VAR bool good;
        ! Transfer has completed its existing release and linear retract.
        ! Use the existing taught HOME as a clear-camera verification/sit position.
        ! A motion fault before this point does NOT trigger automatic recovery movement.
        TPWrite "Transfer done, going HOME";
        MoveJ pHome,v200,fine,tool0;
        v11AtCheckpoint:=TRUE;
        v11ExpectedStep:=checkpointStep; ! Match the next permit to this checkpoint.
        v11Permit:=FALSE; ! Wait for a fresh permit before continuing.
        v11HomeDone:=TRUE;
        TEST checkpointStep
        CASE 1: TPWrite "CHECK 1/6: bottom block on the stack?";
        CASE 2: TPWrite "CHECK 2/6: middle block on the stack?";
        CASE 3: TPWrite "CHECK 3/6: top block on the stack?";
        CASE 4: TPWrite "CHECK 4/6: top block returned?";
        CASE 5: TPWrite "CHECK 5/6: middle block returned?";
        CASE 6: TPWrite "CHECK 6/6: bottom block returned?";
        ENDTEST
        TPWrite "At HOME: waiting for vision check";
        WHILE (v11Permit = FALSE) AND (v11AbortCycle = FALSE) DO
            WaitCommand sb,good;
            IF good AND sb=234 THEN
                ReceiveV11Control;
            ELSEIF good AND sb=236 THEN
                ReceiveV13Target checkpointStep;
            ELSEIF good AND (sb=220 OR sb=222) THEN
                ! V13.6: a repeated plan (PC resend) is not for this checkpoint. Read past its fields so
                ! none of them is mistaken for a new command start byte.
                TPWrite "PLAN IGNORED here (PC resend)";
                SkipFrame sb+1; ! Read past this packet so its fields cannot become commands.
            ENDIF
        ENDWHILE
        IF (v11AbortCycle = FALSE) THEN
            v11AtCheckpoint:=FALSE;
        ENDIF
    ENDPROC

    ! Control frame 234: 1 HOLD (block further transfers), 2 HOME (only from a checkpoint),
    ! 3 RESET after physical recovery (no motion), 4 PERMIT the next transfer.
    PROC ReceiveV11Control()
        VAR num b{4};
        VAR num ending;
        VAR num k;
        VAR bool good;
        FOR k FROM 1 TO 4 DO
            CaptureByte b{k},good;
            IF (good = FALSE) THEN
                TPWrite "CONTROL IGNORED: frame incomplete";
                RETURN;
            ENDIF
        ENDFOR
        CaptureByte ending,good;
        IF (good = FALSE) OR ending<>235 THEN
            TPWrite "CONTROL IGNORED: wrong end value";
            RETURN;
        ENDIF
        AwaitValue 0,good;
        IF (good = FALSE) THEN
            RETURN;
        ENDIF
        IF b{4}<>((b{1}+b{2}+b{3}) MOD 251)+1 THEN
            TPWrite "CONTROL IGNORED: bad checksum";
            RETURN;
        ENDIF
        IF b{1}<1 OR b{1}>127 THEN
            TPWrite "CONTROL IGNORED: bad token";
            RETURN;
        ENDIF
        IF b{2}<1 OR b{2}>4 OR b{3}<1 OR b{3}>6 THEN
            TPWrite "CONTROL IGNORED: bad action or step";
            RETURN;
        ENDIF
        TEST b{2}
        CASE 1: TPWrite "PC -> HOLD (cycle "+NumToStr(b{1},0)+")";
        CASE 2: TPWrite "PC -> go HOME (cycle "+NumToStr(b{1},0)+")";
        CASE 3: TPWrite "PC -> RESET (cycle "+NumToStr(b{1},0)+")";
        CASE 4: TPWrite "PC -> CONTINUE after check "+NumToStr(b{3},0)+"/6";
        ENDTEST
        ! Only operator HOLD/RESET may reconcile an unknown PC token. Neither moves.
        ! Motion HOME and automatic PERMIT always require the exact active token.
        IF (b{2}=2 OR b{2}=4) AND b{1}<>v10Token THEN
            TPWrite "CONTROL IGNORED: not for this cycle";
            RETURN;
        ENDIF
        IF b{2}=4 THEN
            IF enableMotion AND v11AtCheckpoint AND (v11FaultLatch = FALSE) AND b{3}=v11ExpectedStep THEN
                v11Permit:=TRUE;
                TPWrite "CAMERA OK: continue after step" \Num:=v11ExpectedStep;
            ELSE
                TPWrite "PERMIT IGNORED: not waiting at that step";
            ENDIF
        ELSEIF b{2}=1 THEN
            v11FaultLatch:=TRUE;
            v11Permit:=FALSE; ! Wait for a fresh permit before continuing.
            v10Phase:=5;
            TPWrite "HOLD: next transfer blocked";
            TPWrite "HOLD: gripper left as it is, no HOME";
        ELSEIF b{2}=2 THEN
            IF (enableMotion = FALSE) THEN
                TPWrite "HOME REFUSED: enableMotion is FALSE";
                RETURN;
            ENDIF
            v11FaultLatch:=TRUE;
            v11Permit:=FALSE; ! Wait for a fresh permit before continuing.
            v10Phase:=5;
            IF v11AtCheckpoint THEN
                IF (v11HomeDone = FALSE) THEN
                    ! Called only AFTER the existing place/retract at a taught approach.
                    ! Original pHome, speed, orientation and tool are unchanged.
                    MoveJ pHome,v200,fine,tool0;
                    v11HomeDone:=TRUE;
                ENDIF
                TPWrite "HOME reached, cycle held for recovery";
            ELSE
                TPWrite "HOME REFUSED: not at a checkpoint";
                TPWrite "Recover by hand on the pendant";
            ENDIF
        ELSEIF b{2}=3 THEN
            IF dynamicStage<>0 OR buildCount<>0 OR planCount<>0 THEN
                TPWrite "RESET REFUSED: static or test session open";
                RETURN;
            ENDIF
            ! Handler executes only in receiver/checkpoint, not during a motion instruction.
            ! PC sends action3 only after explicit operator physical-recovery acknowledgement.
            ! Explicit operator recovery acknowledgement; reset itself sends NO motion/I/O.
            v11AbortCycle:=TRUE;
            v11FaultLatch:=FALSE;
            v11Permit:=FALSE; ! Wait for a fresh permit before continuing.
            v11ExpectedStep:=0;
            v10Phase:=0;
            v10Token:=b{1};
            TPWrite "RESET done (no motion)";
            TPWrite "Confirm RESET COMPLETE on the PC";
        ELSE
            TPWrite "CONTROL IGNORED: unknown action";
        ENDIF
    ENDPROC

    ! Camera target frame 236, accepted only at the matching checkpoint with the active token.
    ! Mode 1 RE-PICK: the transfer did not happen. Steps 1..3 pick again at camera XY and place on
    ! the same level; steps 4..6 take the same level again and drop at camera XY. Then HOME, same step.
    ! Mode 2 NEW TARGET: the next block moved. Store its XY and continue as a normal permit.
    PROC ReceiveV13Target(num atStep)
        VAR num b{9};
        VAR num ending;
        VAR num k;
        VAR num level;
        VAR num tx;
        VAR num ty;
        VAR num sum;
        VAR bool good;
        VAR robtarget pAbove;
        VAR robtarget pGrip;
        VAR robtarget sAbove;
        VAR robtarget sGrip;
        FOR k FROM 1 TO 9 DO
            CaptureByte b{k},good;
            IF (good = FALSE) THEN
                TPWrite "CAMERA TARGET IGNORED: incomplete";
                RETURN;
            ENDIF
        ENDFOR
        CaptureByte ending,good;
        IF (good = FALSE) OR ending<>237 THEN
            TPWrite "CAMERA TARGET IGNORED: wrong end value";
            RETURN;
        ENDIF
        AwaitValue 0,good;
        IF (good = FALSE) THEN
            RETURN;
        ENDIF
        sum:=0;
        FOR k FROM 1 TO 8 DO
            sum:=sum+b{k};
        ENDFOR
        IF b{9}<>(sum MOD 251)+1 THEN
            TPWrite "CAMERA TARGET IGNORED: bad checksum";
            RETURN;
        ENDIF
        IF b{1}<>v10Token OR b{2}<1 OR b{2}>2 OR b{3}<1 OR b{3}>6 OR b{4}<1 OR b{4}>127 THEN
            TPWrite "CAMERA TARGET IGNORED: not for this cycle";
            RETURN;
        ENDIF
        IF b{5}>16 OR b{6}>128 OR b{7}>16 OR b{8}>128 THEN
            TPWrite "CAMERA TARGET IGNORED: bad XY bytes";
            RETURN;
        ENDIF
        tx:=100+((b{5}-1)*128+b{6}-1)/2;
        ty:=-400+((b{7}-1)*128+b{8}-1)/2;
        IF b{2}=1 THEN
            TPWrite "PC -> RE-PICK at check "+NumToStr(b{3},0)+"/6: "+XYText(tx,ty);
        ELSE
            TPWrite "PC -> NEXT BLOCK MOVED to "+XYText(tx,ty)+", then continue";
        ENDIF
        IF atStep=0 OR (v11AtCheckpoint=FALSE) OR b{3}<>atStep OR b{3}<>v11ExpectedStep THEN
            TPWrite "CAMERA TARGET IGNORED: not waiting there";
            RETURN;
        ENDIF
        IF v11FaultLatch OR (enableMotion=FALSE) OR (enableDynamicCycle=FALSE) THEN
            TPWrite "CAMERA TARGET IGNORED: hold or motion off";
            RETURN;
        ENDIF
        IF b{4}=v13LastAttempt THEN
            TPWrite "CAMERA TARGET IGNORED: already done";
            RETURN;
        ENDIF
        IF b{2}=2 AND (atStep=3 OR atStep=6) THEN
            TPWrite "CAMERA TARGET IGNORED: no next transfer";
            RETURN;
        ENDIF
        tx:=100+((b{5}-1)*128+b{6}-1)/2;
        ty:=-400+((b{7}-1)*128+b{8}-1)/2;
        IF V10PointOK(tx,ty)=FALSE THEN
            TPWrite "CAMERA TARGET REJECTED: XY not reachable";
            TPWrite "BAD X" \Num:=tx;
            TPWrite "BAD Y" \Num:=ty;
            RETURN;
        ENDIF
        v13LastAttempt:=b{4};
        IF b{2}=2 THEN
            IF atStep<3 THEN
                v10PickX{atStep+1}:=tx;v10PickY{atStep+1}:=ty;
                TPWrite "CAMERA: new pickup XY for level" \Num:=atStep+1;
            ELSE
                v10DropX{6-atStep}:=tx;v10DropY{6-atStep}:=ty;
                TPWrite "CAMERA: new drop XY for level" \Num:=6-atStep;
            ENDIF
            TPWrite "CAMERA OK: continue after step" \Num:=v11ExpectedStep;
            v11Permit:=TRUE;
            RETURN;
        ENDIF
        pAbove:=pP7Above;
        pGrip:=pP7Pick;
        pAbove.trans.x:=tx;pAbove.trans.y:=ty;
        pGrip.trans.x:=tx;pGrip.trans.y:=ty;
        TPWrite "CAMERA RE-PICK, attempt" \Num:=b{4};
        v11AtCheckpoint:=FALSE;
        IF atStep<=3 THEN
            level:=atStep;
            v10PickX{level}:=tx;v10PickY{level}:=ty;
            GetStackTargets stackSite,level,sAbove,sGrip;
            TPWrite "RE-PICK: picking again for level" \Num:=level;
            V10Transfer pAbove,pGrip,sAbove,sGrip;
        ELSE
            level:=7-atStep;
            v10DropX{level}:=tx;v10DropY{level}:=ty;
            ReturnTargets tx,ty,pAbove,pGrip;
            GetStackTargets stackSite,level,sAbove,sGrip;
            TPWrite "RE-PICK: block still on stack, level" \Num:=level;
            V10Transfer sAbove,sGrip,pAbove,pGrip;
        ENDIF
        TPWrite "RE-PICK done, going HOME";
        MoveJ pHome,v200,fine,tool0;
        v11AtCheckpoint:=TRUE;
        v11HomeDone:=TRUE;
        v11Permit:=FALSE; ! Wait for a fresh permit before continuing.
        TPWrite "At HOME: camera checks the same step";
    ENDPROC

    ! ------------------------------------------------------------
    ! SECTION 4  TEST PAGE (single supervised pick and test stack)
    ! ------------------------------------------------------------

    ! TEST page frame 232 (wide 0.5 mm XY) or the older 230 (0.1 mm XY).
    ! Action 1 HOVER, 2 PICK, 3 REPLACE or CANCEL, 4 HOVER + PICK, 5..7 build stack level 1..3,
    ! 8 return all built blocks. PICK and REPLACE must repeat the locked token and XY bytes.
    PROC ReceiveDynamicTest(num wide)
        VAR num b{7};
        VAR num ending := 0;
        VAR num i;
        VAR num check := 0;
        VAR num x;
        VAR num y;
        VAR bool good := FALSE;
        check := 0;
        ending := 0;
        good := FALSE;
        FOR i FROM 1 TO 7 DO
            CaptureByte b{i}, good;
            IF (good = FALSE) THEN
                TPWrite "DYNAMIC FRAME TIMEOUT - NO MOVE";
                RETURN;
            ENDIF
        ENDFOR
        CaptureByte ending, good;
        IF good = FALSE OR (wide = 0 AND ending <> 231) OR (wide = 1 AND ending <> 233) THEN
            TPWrite "DYNAMIC BAD END - NO MOVE";
            RETURN;
        ENDIF
        AwaitValue 0, good;
        IF (good = FALSE) THEN
            RETURN;
        ENDIF
        FOR i FROM 1 TO 6 DO
            check := check + b{i};
        ENDFOR
        IF b{7} <> (check MOD 251) + 1 THEN
            TPWrite "DYNAMIC BAD CHECKSUM - NO MOVE";
            RETURN;
        ENDIF
        IF b{1} < 1 OR b{1} > 127 OR b{2} < 1 OR b{2} > 3 THEN
            TPWrite "DYNAMIC BAD TOKEN/ACTION";
            RETURN;
        ENDIF
        IF b{3} < 1 OR b{3} > 16 OR b{5} < 1 OR b{5} > 16 THEN
            RETURN;
        ENDIF
        IF b{4} < 1 OR b{4} > 128 OR b{6} < 1 OR b{6} > 128 THEN
            RETURN;
        ENDIF
        IF wide = 1 THEN
            ! 0.5 mm resolution: X = 100 + q/2, Y = -400 + q/2
            x := 100 + ((b{3}-1)*128 + b{4}-1)/2;
            y := -400 + ((b{5}-1)*128 + b{6}-1)/2;
            IF x < dynamicWideMinX OR Sqrt(x*x + y*y) < dynamicReachMin OR Sqrt(x*x + y*y) > dynamicReachMax THEN
                TPWrite "XY OUTSIDE REACHABLE TEST AREA - NO MOVE";
                TPWrite "X" \Num:=x;
                TPWrite "Y" \Num:=y;
                RETURN;
            ENDIF
        ELSE
            x := 350 + ((b{3}-1)*128 + b{4}-1)/10;
            y := -100 + ((b{5}-1)*128 + b{6}-1)/10;
            IF x < dynamicMinX OR x > dynamicMaxX OR y < dynamicMinY OR y > dynamicMaxY THEN
                TPWrite "DYNAMIC XY OUTSIDE INNER TEST REGION";
                RETURN;
            ENDIF
        ENDIF
        IF V10PointOK(x,y) = FALSE THEN
            TPWrite "TEST XY IN RESTRICTED AREA - NO MOVE";
            RETURN;
        ENDIF
        IF uiMode<>0 AND uiMode<>3 THEN
            TPWrite "TEST REJECTED: SELECT TEST MODE";
            ReportSession;
            RETURN;
        ENDIF
        TEST b{2}
        CASE 1: TPWrite "PC -> TEST HOVER above "+XYText(x,y);
        CASE 2: TPWrite "PC -> TEST PICK at "+XYText(x,y);
        CASE 3: TPWrite "PC -> TEST REPLACE / CANCEL";
        ENDTEST
        IF v10Phase = 1 OR v10Phase = 2 OR v10Phase = 3 THEN
            ReportSession;
            TPWrite "V10 SESSION ACTIVE - TEST BLOCKED";
            RETURN;
        ENDIF
        IF enableMotion = FALSE THEN
            TPWrite "DYNAMIC LOCKED - enableMotion IS FALSE";
            RETURN;
        ENDIF
        IF enableDynamicTest = FALSE THEN
            TPWrite "DYNAMIC LOCKED - enableDynamicTest IS FALSE";
            RETURN;
        ENDIF
        IF planCount <> 0 THEN
            TPWrite "DYNAMIC LOCKED - STACK PLAN ACTIVE";
            RETURN;
        ENDIF
        IF b{2} >= 5 THEN
            ! 5,6,7 = pick block at XY and stack at S1 level 1,2,3 ; 8 = return all blocks to where they were picked.
            IF dynamicStage <> 0 THEN
                TPWrite "BUILD REJECTED - HOVER SESSION ACTIVE";
                RETURN;
            ENDIF
            IF b{2} <= 7 THEN
                TPWrite "BUILD FRAME LEVEL" \Num:=b{2} - 4;
                IF b{2} - 4 > buildCount + 1 THEN
                    TPWrite "BUILD WRONG LEVEL - NO MOVE";
                    TPWrite "EXPECTED LEVEL" \Num:=buildCount + 1;
                    RETURN;
                ENDIF
                IF b{2} - 4 <= buildCount THEN
                    TPWrite "REDOING LEVEL (OPERATOR RETRY)" \Num:=b{2} - 4;
                ENDIF
                dynamicStage := -1;
                BuildPickToStack b{2} - 4, x, y;
                buildX{b{2} - 4} := x;
                buildY{b{2} - 4} := y;
                buildCount := b{2} - 4;
                dynamicStage := 0;
                TPWrite "STACKED LEVEL" \Num:=buildCount;
                IF buildCount >= 3 THEN
                    TPWrite "STACK FULL - HOME - PRESS RETURN ALL";
                ELSE
                    TPWrite "HOME - SELECT NEXT BLOCK";
                ENDIF
            ELSE
                IF buildCount < 1 THEN
                    TPWrite "NOTHING TO RETURN - NO MOVE";
                    RETURN;
                ENDIF
                dynamicStage := -1;
                ReturnAllFromStack;
                dynamicStage := 0;
                TPWrite "ALL BLOCKS RETURNED - HOME";
            ENDIF
            RETURN;
        ENDIF
        IF b{2} = 1 OR b{2} = 4 THEN
            IF dynamicStage <> 0 THEN
                TPWrite "DYNAMIC SESSION ACTIVE - NO NEW HOVER";
                RETURN;
            ENDIF
            dynamicToken := b{1};
            dynamicX := x;
            dynamicY := y;
            dynamicWireMode := wide;
            FOR i FROM 1 TO 4 DO
                dynamicWireXY{i} := b{i+2};
            ENDFOR
            TPWrite "LOCKED TEST TOKEN" \Num:=dynamicToken;
            dynamicAbove := pP7Above;
            dynamicGrip := pP7Pick;
            dynamicAbove.trans.x := x;
            dynamicAbove.trans.y := y;
            dynamicGrip.trans.x := x;
            dynamicGrip.trans.y := y;
            TPWrite "DYNAMIC HOVER XY" \Num:=x;
            TPWrite "DYNAMIC Y" \Num:=y;
            dynamicStage := -1;
            TraceMotion "DYNAMIC HOVER - VIA HOME";
            MoveJ pHome,v200,fine,tool0;
            TraceMotion "DYNAMIC HOVER - APPROACH";
            MoveJ dynamicAbove,v200,fine,tool0;
            OpenGripper;
            IF b{2} = 4 THEN
                ! GO + PICK: XY was locked by the camera BEFORE motion; no further camera input needed.
                TraceMotion "GO+PICK - DESCEND AND LIFT";
                MoveL dynamicGrip,v150,fine,tool0;
                CloseGripper;
                MoveL dynamicAbove,v150,fine,tool0;
                dynamicStage := 2;
                TPWrite "PICK/LIFT COMPLETE - REQUEST REPLACE";
            ELSE
                dynamicStage := 1;
                TPWrite "HOVER COMPLETE - VISUALLY CHECK ALIGNMENT";
            ENDIF
        ELSE
            IF b{1} <> dynamicToken THEN
                TPWrite "DYNAMIC TOKEN MISMATCH - NO MOVE";
                TPWrite "EXPECTED TOKEN" \Num:=dynamicToken;
                TPWrite "RECEIVED TOKEN" \Num:=b{1};
                TPWrite "STORED STAGE" \Num:=dynamicStage;
                RETURN;
            ENDIF
            IF wide <> dynamicWireMode THEN
                TPWrite "TEST FRAME MODE MISMATCH - NO MOVE";
                RETURN;
            ENDIF
            FOR i FROM 1 TO 4 DO
                IF b{i+2} <> dynamicWireXY{i} THEN
                    TPWrite "DYNAMIC XY BYTE MISMATCH - NO MOVE";
                    TPWrite "BYTE INDEX" \Num:=i;
                    TPWrite "EXPECTED BYTE" \Num:=dynamicWireXY{i};
                    TPWrite "RECEIVED BYTE" \Num:=b{i+2};
                    TPWrite "EXPECTED X" \Num:=dynamicX;
                    TPWrite "RECEIVED X" \Num:=x;
                    TPWrite "EXPECTED Y" \Num:=dynamicY;
                    TPWrite "RECEIVED Y" \Num:=y;
                    RETURN;
                ENDIF
            ENDFOR
            ! Rebuild from the locked session after a task restart. Never adopt new follow-up XY.
            dynamicAbove := pP7Above;
            dynamicGrip := pP7Pick;
            dynamicAbove.trans.x := dynamicX;
            dynamicAbove.trans.y := dynamicY;
            dynamicGrip.trans.x := dynamicX;
            dynamicGrip.trans.y := dynamicY;
            IF b{2} = 2 AND dynamicStage = 1 THEN
                dynamicStage := -1;
                TraceMotion "DYNAMIC PICK - DESCEND AND LIFT";
                OpenGripper;
                MoveL dynamicGrip,v150,fine,tool0;
                CloseGripper;
                MoveL dynamicAbove,v150,fine,tool0;
                dynamicStage := 2;
                TPWrite "PICK/LIFT COMPLETE - REQUEST REPLACE";
            ELSEIF b{2} = 2 AND dynamicStage = 2 THEN
                TPWrite "DUPLICATE PICK - ALREADY LIFTED - NO MOVE";
            ELSEIF b{2} = 3 AND dynamicStage = 2 THEN
                dynamicStage := -1;
                TraceMotion "DYNAMIC REPLACE - ORIGINAL XY";
                MoveL dynamicGrip,v150,fine,tool0;
                OpenGripper;
                MoveL dynamicAbove,v150,fine,tool0;
                MoveJ pHome,v200,fine,tool0;
                dynamicStage := 0;
                TPWrite "DYNAMIC REPLACED - HOME COMPLETE";
            ELSEIF b{2} = 3 AND dynamicStage = 1 THEN
                dynamicStage := -1;
                TraceMotion "DYNAMIC CANCEL - HOME";
                MoveJ pHome,v200,fine,tool0;
                dynamicStage := 0;
                TPWrite "DYNAMIC HOVER CANCELLED - HOME COMPLETE";
            ELSEIF b{2} = 3 AND dynamicStage = 0 THEN
                TPWrite "DUPLICATE HOME - SESSION ALREADY CLOSED";
            ELSE
                TPWrite "DYNAMIC WRONG STAGE - NO MOVE";
                TPWrite "STAGE" \Num:=dynamicStage;
                TPWrite "ACTION" \Num:=b{2};
            ENDIF
        ENDIF
    ENDPROC

    ! TEST build: pick the block at camera XY and place it on S1 at `level`, then HOME.
    PROC BuildPickToStack(num level, num x, num y)
        VAR robtarget bAbove;
        VAR robtarget bGrip;
        VAR robtarget sAbove;
        VAR robtarget sLevel;
        bAbove := pP7Above;
        bGrip := pP7Pick;
        bAbove.trans.x := x;
        bAbove.trans.y := y;
        bGrip.trans.x := x;
        bGrip.trans.y := y;
        GetStackTargets 1, level, sAbove, sLevel;
        TPWrite "BUILD PICK X" \Num:=x;
        TPWrite "BUILD PICK Y" \Num:=y;
        TPWrite "BUILD TO STACK LEVEL" \Num:=level;
        TraceMotion "BUILD - VIA HOME";
        MoveJ pHome,v200,fine,tool0;
        TraceMotion "BUILD - APPROACH BLOCK";
        MoveJ bAbove,v200,fine,tool0;
        OpenGripper;
        TraceMotion "BUILD - DESCEND";
        MoveL bGrip,v150,fine,tool0;
        CloseGripper;
        TraceMotion "BUILD - LIFT";
        MoveL bAbove,v150,fine,tool0;
        TraceMotion "BUILD - TO STACK";
        MoveJ sAbove,v200,fine,tool0;
        MoveL sLevel,v150,fine,tool0;
        OpenGripper;
        TraceMotion "BUILD - RETRACT";
        MoveL sAbove,v150,fine,tool0;
        MoveJ pHome,v200,fine,tool0;
    ENDPROC

    ! TEST build: unstack top first; each block goes back to the XY it was picked from.
    PROC ReturnAllFromStack()
        VAR num l;
        VAR robtarget oAbove;
        VAR robtarget oGrip;
        VAR robtarget sAbove;
        VAR robtarget sLevel;
        FOR l FROM buildCount TO 1 STEP -1 DO
            GetStackTargets 1, l, sAbove, sLevel;
            ReturnTargets buildX{l},buildY{l},oAbove,oGrip;
            TPWrite "UNSTACK LEVEL" \Num:=l;
            TraceMotion "RETURN - TO STACK";
            MoveJ sAbove,v200,fine,tool0;
            OpenGripper;
            MoveL sLevel,v150,fine,tool0;
            CloseGripper;
            MoveL sAbove,v150,fine,tool0;
            TraceMotion "RETURN - TO ORIGINAL XY";
            MoveJ oAbove,v200,fine,tool0;
            MoveL oGrip,v150,fine,tool0;
            OpenGripper;
            MoveL oAbove,v150,fine,tool0;
        ENDFOR
        MoveJ pHome,v200,fine,tool0;
        buildCount := 0;
    ENDPROC

    ! ------------------------------------------------------------
    ! SECTION 5  STATIC MODE (taught positions P1..P9)
    ! ------------------------------------------------------------

    ! Static plan fields: site 161..163, L1 177..185, L2 193..201, L3 209..217. Pickups must differ.
    FUNC bool DecodePlan()
        IF rawStack < 161 OR rawStack > 163 THEN
            RETURN FALSE;
        ENDIF
        IF rawL1 < 177 OR rawL1 > 185 THEN
            RETURN FALSE;
        ENDIF
        IF rawL2 < 193 OR rawL2 > 201 THEN
            RETURN FALSE;
        ENDIF
        IF rawL3 < 209 OR rawL3 > 217 THEN
            RETURN FALSE;
        ENDIF
        stackNo := rawStack - 160;
        pick1 := rawL1 - 176;
        pick2 := rawL2 - 192;
        pick3 := rawL3 - 208;
        IF pick1 = pick2 OR pick1 = pick3 OR pick2 = pick3 THEN
            RETURN FALSE;
        ENDIF
        RETURN TRUE;
    ENDFUNC

    ! Static return fields for the current site: L3 17..25, L2 33..41, L1 49..57. Must differ.
    FUNC bool DecodeReturn()
        IF rawRS <> 128 + stackNo THEN
            RETURN FALSE;
        ENDIF
        IF rawR3 < 17 OR rawR3 > 25 THEN
            RETURN FALSE;
        ENDIF
        IF rawR2 < 33 OR rawR2 > 41 THEN
            RETURN FALSE;
        ENDIF
        IF rawR1 < 49 OR rawR1 > 57 THEN
            RETURN FALSE;
        ENDIF
        ret3 := rawR3 - 16;
        ret2 := rawR2 - 32;
        ret1 := rawR1 - 48;
        IF ret1 = ret2 OR ret1 = ret3 OR ret2 = ret3 THEN
            RETURN FALSE;
        ENDIF
        RETURN TRUE;
    ENDFUNC

    ! A stack site is accepted only once until it has been returned and released.
    FUNC bool StackAlreadyStored(num requestedStack)
        VAR num k;
        IF requestedStack < 1 OR requestedStack > 3 THEN
            RETURN TRUE;
        ENDIF
        IF planCount < 0 OR planCount > 3 THEN
            RETURN TRUE;
        ENDIF
        IF planCount = 0 THEN
            RETURN FALSE;
        ENDIF
        FOR k FROM 1 TO planCount DO
            IF stackOrder{k} = requestedStack THEN
                RETURN TRUE;
            ENDIF
        ENDFOR
        RETURN FALSE;
    ENDFUNC

    ! FALSE when a requested pickup is already used by another stored plan.
    FUNC bool SourceAvailable()
        VAR num k;
        VAR num l;
        VAR num s;
        IF planCount < 0 OR planCount > 3 THEN
            RETURN FALSE;
        ENDIF
        IF planCount = 0 THEN
            RETURN TRUE;
        ENDIF
        FOR k FROM 1 TO planCount DO
            s := stackOrder{k};
            IF s < 1 OR s > 3 THEN
                RETURN FALSE;
            ENDIF
            FOR l FROM 1 TO 3 DO
                IF storedPick{s,l} = pick1 OR storedPick{s,l} = pick2 OR storedPick{s,l} = pick3 THEN
                    RETURN FALSE;
                ENDIF
            ENDFOR
        ENDFOR
        RETURN TRUE;
    ENDFUNC

    ! Records the decoded static plan for its stack site.
    PROC StorePlan()
        IF planCount < 0 OR planCount >= 3 OR stackNo < 1 OR stackNo > 3 THEN
            TPWrite "INVALID PLAN COUNT / STACK - NO ARRAY WRITE";
            Stop;
            RETURN;
        ENDIF
        planCount := planCount + 1;
        stackOrder{planCount} := stackNo;
        storedPick{stackNo,1} := pick1;
        storedPick{stackNo,2} := pick2;
        storedPick{stackNo,3} := pick3;
    ENDPROC

    ! Static: build the stack bottom (L1) to top (L3), then HOME.
    PROC BuildOneStack(num s)
        VAR num l;
        IF s < 1 OR s > 3 THEN
            TPWrite "INVALID STACK INDEX";
            Stop;
            RETURN;
        ENDIF
        FOR l FROM 1 TO 3 DO
            IF storedPick{s,l} < 1 OR storedPick{s,l} > 9 THEN
                TPWrite "INVALID STORED PICKUP";
                Stop;
                RETURN;
            ENDIF
            TPWrite "NOW RUNNING STACK S" \Num:=s;
            TPWrite "STACK LEVEL" \Num:=l;
            Transfer storedPick{s,l},s,l,FALSE;
        ENDFOR
        TraceMotion "MOVE HOME - START";
        MoveJ pHome,v200,fine,tool0;
        TraceMotion "MOVE HOME - COMPLETE";
    ENDPROC

    ! Static: wait (5 command attempts) for a return frame 250; good = TRUE only when valid.
    PROC ReceiveReturnPlan(VAR bool good)
        VAR num t;
        VAR bool ok := FALSE;
        VAR num sb := 0;
        VAR num eb := 0;
        good := FALSE;
        rawRS := 0;
        rawR3 := 0;
        rawR2 := 0;
        rawR1 := 0;
        FOR t FROM 1 TO 5 DO
            sb := 0;
            CaptureByte sb, ok;
            IF ok AND sb>=226 AND sb<=229 THEN
                SelectDashboard sb;
                TPWrite "STATIC RETURN STILL REQUIRED: 250";
            ELSEIF ok AND sb = 250 THEN
                TPWrite "RETURN FRAME START 250";
                CaptureByte rawRS, ok;
                IF ok THEN
                    CaptureByte rawR3, ok;
                ENDIF
                IF ok THEN
                    CaptureByte rawR2, ok;
                ENDIF
                IF ok THEN
                    CaptureByte rawR1, ok;
                ENDIF
                IF ok THEN
                    CaptureByte eb, ok;
                ENDIF
                IF ok AND eb = 251 THEN
                    good := DecodeReturn();
                ENDIF
                IF (good = FALSE) THEN
                    TPWrite "RETURN FRAME REJECTED";
                    TPWrite "RETURN STACK" \Num:=rawRS;
                    TPWrite "RETURN L3" \Num:=rawR3;
                    TPWrite "RETURN L2" \Num:=rawR2;
                    TPWrite "RETURN L1" \Num:=rawR1;
                    TPWrite "RETURN END" \Num:=eb;
                ENDIF
                AwaitValue 0, ok;
                RETURN;
            ENDIF
        ENDFOR
    ENDPROC

    ! Static: after stacking, wait for the return plan and take the stack down.
    PROC HandleReturn(num s)
        VAR bool good := FALSE;
        ! BuildOneStack has completed its fine move to home before this call.
        WHILE (good = FALSE) DO
            TPWrite "HOME - LISTENING FOR AUTOMATIC RETURN PLAN";
            ReceiveReturnPlan good;
        ENDWHILE
        returnP{s,1} := ret1;
        returnP{s,2} := ret2;
        returnP{s,3} := ret3;
        TPWrite "RETURNING TOP FIRST TO CAMERA EMPTY POSITIONS";
        ReturnOneStack s;
        ReleaseStack s;
        TPWrite "STATIC PLANS REMAINING" \Num:=planCount;
        TPWrite "RETURN COMPLETE - HOME - READY FOR NEXT STACK";
    ENDPROC

    ! Static: unstack top first; each block goes to its return position P.
    PROC ReturnOneStack(num s)
        VAR num l;
        IF s < 1 OR s > 3 THEN
            TPWrite "INVALID STACK INDEX";
            Stop;
            RETURN;
        ENDIF
        FOR l FROM 3 TO 1 STEP -1 DO
            IF returnP{s,l} < 1 OR returnP{s,l} > 9 THEN
                TPWrite "INVALID RETURN POSITION";
                Stop;
                RETURN;
            ENDIF
            TPWrite "RETURNING STACK S" \Num:=s;
            TPWrite "RETURN LEVEL" \Num:=l;
            Transfer returnP{s,l},s,l,TRUE;
        ENDFOR
        TraceMotion "MOVE HOME - START";
        MoveJ pHome,v200,fine,tool0;
        TraceMotion "MOVE HOME - COMPLETE";
    ENDPROC

    ! Static: forget a returned stack so the site and positions can be planned again.
    PROC ReleaseStack(num s)
        VAR num k;
        VAR num j := 0;
        VAR bool found := FALSE;
        FOR k FROM 1 TO planCount DO
            IF stackOrder{k} = s THEN
                found := TRUE;
            ELSE
                j := j + 1;
                stackOrder{j} := stackOrder{k};
            ENDIF
        ENDFOR
        IF found THEN
            planCount := planCount - 1;
        ENDIF
        FOR k FROM 1 TO 3 DO
            storedPick{s,k} := 0;
            returnP{s,k} := 0;
        ENDFOR
    ENDPROC

    ! Static: returning FALSE picks from P and places on the stack; TRUE does the reverse.
    PROC Transfer(num positionNo, num destinationStack, num level, bool returning)
        VAR robtarget pAbove;
        VAR robtarget pPick;
        VAR robtarget stackAbove;
        VAR robtarget stackLevel;
        GetPickupTargets positionNo, pAbove, pPick;
        GetStackTargets destinationStack, level, stackAbove, stackLevel;
        TPWrite "STACK SITE COORDINATES S" \Num:=stackSite;
        IF returning THEN
            TPWrite "DROP AT P" \Num:=positionNo;
            TPWrite "FROM STACK S" \Num:=destinationStack;
            TPWrite "FROM LEVEL" \Num:=level;
            PickAndPlace stackAbove,stackLevel,pAbove,pPick;
        ELSE
            TPWrite "PICK FROM P" \Num:=positionNo;
            TPWrite "PLACE STACK S" \Num:=destinationStack;
            TPWrite "PLACE LEVEL" \Num:=level;
            PickAndPlace pAbove,pPick,stackAbove,stackLevel;
        ENDIF
    ENDPROC

    ! ------------------------------------------------------------
    ! SECTION 6  TARGETS, MOTION AND GRIPPER
    ! ------------------------------------------------------------

    ! Taught approach and grip targets for pickup position P1..P9.
    ! Drop targets for a block returned to the table at XY (x, y). Orientation and configuration come
    ! from P7, like every camera target. Z comes from returnDropZ / returnAboveZ (section B) when they
    ! are within the limits in section C, otherwise from the taught P7 heights.
    PROC ReturnTargets(num x, num y, VAR robtarget above, VAR robtarget drop)
        above := pP7Above; ! Copy the taught approach height and orientation first.
        drop := pP7Pick; ! Keep the taught grip orientation for the return drop.
        above.trans.x := x;
        above.trans.y := y;
        drop.trans.x := x;
        drop.trans.y := y;
        IF returnDropZ >= pP7Pick.trans.z + returnDropMinDZ AND returnDropZ <= pP7Pick.trans.z + returnDropMaxDZ
            AND returnAboveZ >= returnDropZ + returnAboveMinGap AND returnAboveZ <= startupSafeZ THEN
            drop.trans.z := returnDropZ; ! Apply the release height after checking its limits.
            above.trans.z := returnAboveZ;
        ELSE
            TPWrite "RETURN Z out of range: using taught P7 heights";
        ENDIF
    ENDPROC

    PROC GetPickupTargets(num p, VAR robtarget above, VAR robtarget pick)
        TEST p
        CASE 1:
            above := pP1Above;
            pick := pP1Pick;
        CASE 2:
            above := pP2Above;
            pick := pP2Pick;
        CASE 3:
            above := pP3Above;
            pick := pP3Pick;
        CASE 4:
            above := pP4Above;
            pick := pP4Pick;
        CASE 5:
            above := pP5Above;
            pick := pP5Pick;
        CASE 6:
            above := pP6Above;
            pick := pP6Pick;
        CASE 7:
            above := pP7Above; ! Copy the taught approach height and orientation first.
            pick := pP7Pick;
        CASE 8:
            above := pP8Above;
            pick := pP8Pick;
        CASE 9:
            above := pP9Above;
            pick := pP9Pick;
        DEFAULT:
            TPWrite "INVALID PICKUP";
            Stop;
        ENDTEST
    ENDPROC

    ! Approach and level targets for the stack site (stackSite, or the site fixed for this cycle).
    PROC GetStackTargets(num s, num level, VAR robtarget above, VAR robtarget place)
        VAR num site;
        ! s is the requested stack (kept for bookkeeping); the coordinates come from stackSite.
        site := stackSite;
        IF v10Phase=1 OR v10Phase=2 OR v10Phase=3 THEN
            site := v10Site; ! Keep the site chosen when this cycle began.
        ENDIF
        IF site < 1 OR site > 3 THEN
            site := 1;
        ENDIF
        TEST site
        CASE 1:
            above := pS1Above;
            TEST level
            CASE 1: place := pS1Level1;
            CASE 2: place := pS1Level2;
            CASE 3: place := pS1Level3;
            DEFAULT: Stop;
            ENDTEST
        CASE 2:
            above := pS2Above;
            TEST level
            CASE 1: place := pS2Level1;
            CASE 2: place := pS2Level2;
            CASE 3: place := pS2Level3;
            DEFAULT: Stop;
            ENDTEST
        CASE 3:
            above := pS3Above;
            TEST level
            CASE 1: place := pS3Level1;
            CASE 2: place := pS3Level2;
            CASE 3: place := pS3Level3;
            DEFAULT: Stop;
            ENDTEST
        DEFAULT:
            Stop;
        ENDTEST
    ENDPROC

    ! Static transfer with a TraceMotion record before every move.
    PROC PickAndPlace(robtarget fromAbove, robtarget fromGrip,
                      robtarget toAbove, robtarget toRelease)
        ! Motion path and gripper I/O must be checked on actual cell.
        TraceMotion "PICK - MOVE TO APPROACH";
        MoveJ fromAbove, v200, fine, tool0;
        TraceMotion "PICK - OPEN BEFORE DESCENT";
        OpenGripper;
        TraceMotion "PICK - DESCEND";
        MoveL fromGrip, v150, fine, tool0;
        CloseGripper;
        TraceMotion "PICK - RETRACT";
        MoveL fromAbove, v150, fine, tool0;
        TraceMotion "PLACE - MOVE TO APPROACH";
        MoveJ toAbove, v200, fine, tool0;
        TraceMotion "PLACE - DESCEND";
        MoveL toRelease, v150, fine, tool0;
        OpenGripper;
        TraceMotion "PLACE - RETRACT";
        MoveL toAbove, v150, fine, tool0;
    ENDPROC

    ! Records the step name and joint values before a move (Program Data: lastMotionStep).
    PROC TraceMotion(string stepName)
        lastMotionStep := stepName;
        lastMotionJoints := CJointT();
        IF verboseDiagnostics THEN
        TPWrite "A1" \Num:=lastMotionJoints.robax.rax_1;
        TPWrite "A2" \Num:=lastMotionJoints.robax.rax_2;
        TPWrite "A3" \Num:=lastMotionJoints.robax.rax_3;
        TPWrite "A4" \Num:=lastMotionJoints.robax.rax_4;
        TPWrite "A5" \Num:=lastMotionJoints.robax.rax_5;
        TPWrite "A6" \Num:=lastMotionJoints.robax.rax_6;
        ENDIF
        TPWrite stepName;
    ENDPROC

    ! Timed outputs: there is no sensor, so the open state is assumed after 2 s.
    PROC OpenGripper()
        TPWrite "Gripper opening";
        Reset doGripClose; ! Clear the closing output.
        Set doGripOpen;
        WaitTime 2;
        Reset doGripOpen; ! Clear the opening output.
    ENDPROC

    ! Timed outputs: the grip is not confirmed by the robot, only by the PC camera.
    PROC CloseGripper()
        TPWrite "Gripper closing (not confirmed)";
        Reset doGripOpen; ! Clear the opening output.
        Set doGripClose;
        WaitTime 2;
        Reset doGripClose; ! Clear the closing output.
    ENDPROC
ENDMODULE
