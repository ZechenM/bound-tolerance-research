@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
:: Check for description argument
IF "%~1"=="" (
    echo Usage: %0 ^<experiment_description^>
    echo Example: %0 "test_run_1"
    exit /B 1
)

:: Create timestamp-based directory name
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (
    set DATE=%%a%%b
)
for /f "tokens=1-2 delims=: " %%a in ('time /t') do (
    set TIME=%%a%%b
)
SET LOG_DIR=logs\%DATE%_%TIME%_%~1
IF NOT EXIST %LOG_DIR% mkdir %LOG_DIR%

:: Determine script mode
SET SERVER_SCRIPT=server_compressed.py
SET WORKER_SCRIPT=worker_trainer.py
SET SERVER_LOG=%LOG_DIR%\server_dynamic_bound_loss_log.txt
SET WORKER0_LOG=%LOG_DIR%\worker_dynamic_bound_loss_log0.txt
SET WORKER1_LOG=%LOG_DIR%\worker_dynamic_bound_loss_log1.txt
SET WORKER2_LOG=%LOG_DIR%\worker_dynamic_bound_loss_log2.txt

:: Kill any process using port 60000
echo Checking for processes using port 60000...
FOR /F "tokens=5" %%P IN ('netstat -ano ^| findstr :60000') DO (
    echo Killing process %%P using port 60000...
    taskkill /PID %%P /F >NUL 2>&1
    timeout /T 2 >NUL
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

:: Start the server with process ID capture
echo Starting server...
start /B python %SERVER_SCRIPT% > %SERVER_LOG% 2>&1
echo Waiting 5 seconds for server to initialize...
timeout /T 5 >NUL

:: Check if server is running
FOR /F "tokens=5" %%P IN ('netstat -ano ^| findstr :60000') DO (
    echo Server running with PID: %%P
    goto :SERVER_RUNNING
)
echo WARNING: No process found on port 60000! Server may have failed to start.
type %SERVER_LOG%
exit /B 1

:SERVER_RUNNING
:: Check if .server_port file exists (created by Server class on startup)
if exist .server_port (
    echo Server startup confirmed - .server_port file created
) else (
    echo WARNING: .server_port file not found! Server may have failed to start.
    type %SERVER_LOG%
    exit /B 1
)

:: Start workers
echo Starting workers...
start /B python %WORKER_SCRIPT% 0 > %WORKER0_LOG% 2>&1
start /B python %WORKER_SCRIPT% 1 > %WORKER1_LOG% 2>&1
start /B python %WORKER_SCRIPT% 2 > %WORKER2_LOG% 2>&1

:: Monitor logs for completion messages
SET WORKERS_DONE=0
:CHECK_WORKERS
FOR %%F IN ("%WORKER0_LOG%" "%WORKER1_LOG%" "%WORKER2_LOG%") DO (
    FINDSTR /C:"Worker " %%F | FINDSTR /C:"evaluation DONE" >NUL
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