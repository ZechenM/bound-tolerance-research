@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Determine script mode
IF "%1"=="-d" (
    SET SERVER_SCRIPT=server_compressed.py
    SET WORKER_SCRIPT=worker_trainer.py
    SET SERVER_LOG=logs\server_dynamic_bound_loss_log.txt
    SET WORKER0_LOG=logs\worker_dynamic_bound_loss_log0.txt
    SET WORKER1_LOG=logs\worker_dynamic_bound_loss_log1.txt
    SET WORKER2_LOG=logs\worker_dynamic_bound_loss_log2.txt
)

:: Install dependencies
IF EXIST requirements.txt (
    echo Installing required packages...
    pip install -r requirements.txt
) ELSE (
    echo requirements.txt not found. Skipping package installation.
)

:: Create logs directory if not exists
IF NOT EXIST logs mkdir logs

:: Start the server
echo Starting server...
start /B python %SERVER_SCRIPT% > %SERVER_LOG% 2>&1
echo Server started.

:: Start workers
echo Starting workers...
start /B python %WORKER_SCRIPT% 0 > %WORKER0_LOG% 2>&1
start /B python %WORKER_SCRIPT% 1 > %WORKER1_LOG% 2>&1
start /B python %WORKER_SCRIPT% 2 > %WORKER2_LOG% 2>&1

:: Monitor logs for completion messages
SET WORKERS_DONE=0
:CHECK_WORKERS
FOR %%F IN ("%WORKER0_LOG%" "%WORKER1_LOG%" "%WORKER2_LOG%") DO (
    FINDSTR /C:"Worker " %%F | FINDSTR /C:"finished training." >NUL
    IF NOT ERRORLEVEL 1 (
        echo Worker finished: %%F
        SET /A WORKERS_DONE+=1
    )
)

IF %WORKERS_DONE% LSS 3 (
    timeout /T 5 > NUL
    GOTO CHECK_WORKERS
)

echo All workers have finished training. Exiting.
exit /B 0