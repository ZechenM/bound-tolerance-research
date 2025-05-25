#!/bin/bash

# --- Configuration ---
# The log file to monitor.
# You can change this to whatever your log file is named.
LOGFILE="/root/bound-tolerance-research/logs/worker2.log"
# The specific string that indicates training is complete.
DONE_STRING="training DONE"
# --- End Configuration ---

# --- Script Logic ---

# Check if the log file already exists and remove it for a clean start.
# if [ -f "$LOGFILE" ]; then
#   echo "INFO: Found and removed existing log file: $LOGFILE"
#   rm "$LOGFILE"
# fi

echo "----------------------------------------------------"
echo "Log File Monitor Started"
echo "Monitoring File: $LOGFILE"
echo "Timer Starts:   Immediately after file creation"
echo "Trigger for Stop: \"$DONE_STRING\""
echo "----------------------------------------------------"

# Function to get current time in seconds with nanosecond precision
current_time_nanos() {
  date +%s.%N
}

# Initialize timer_started flag and start_time
timer_started=0
start_time=""

# Wait until the log file exists
if [ ! -f "$LOGFILE" ]; then
  echo "Waiting for '$LOGFILE' to be created..."
  # Use inotifywait to wait for the file to be created in the current directory
  # The '-e create' event monitors for file creation.
  # The '--format '%f'' outputs just the filename.
  # We redirect stderr to /dev/null to suppress messages if the directory isn't watched.
  while true; do
    # Monitor for the 'create' event in the current directory ('.')
    # The -q option makes inotifywait quiet.
    # The --format '%f' outputs only the filename of the event.
    # We grep for the exact log file name.
    inotifywait -q -e create --format '%f' /root/bound-tolerance-research/logs 2>/dev/null | grep -q "^${LOGFILE}$" && break
    # Sleep briefly to avoid busy-waiting if inotifywait isn't ideal for some edge case
    sleep 0.5 
  done
  echo "'$LOGFILE' created."
fi

# --- Start the timer immediately after file creation is confirmed ---
start_time=$(current_time_nanos)
timer_started=1 # Set flag to indicate timer has started
echo "----------------------------------"
echo "EVENT: Log file '$LOGFILE' detected/created. Timer started."
echo "Start Time: $(date -d @"${start_time%.*}" +'%Y-%m-%d %H:%M:%S.%N')" # Display start time with nanoseconds
echo "----------------------------------"

echo "Monitoring '$LOGFILE' for completion message..."

# We'll use inotifywait for event-driven monitoring.
while true; do
  # Wait for either a write (close_write) or modification (modify) event on the log file.
  # 'close_write' happens when a file opened for writing is closed.
  # 'modify' happens when the file content is changed.
  # The timeout allows the script to periodically check conditions even if no file event occurs,
  # though for this specific logic (waiting for a string), event-driven is primary.
  inotifywait -q -e close_write -e modify "$LOGFILE" --timeout 5 2>/dev/null

  # Check for the "training DONE" message
  # This check happens after any modification or close_write event, or after a timeout.
  # Use grep -q for a quiet search (no output, just exit status)
  if grep -q "$DONE_STRING" "$LOGFILE"; then
    end_time=$(current_time_nanos)
    echo "----------------------------------"
    echo "EVENT: '$DONE_STRING' detected. Timer stopped."
    echo "End Time: $(date -d @"${end_time%.*}" +'%Y-%m-%d %H:%M:%S.%N')" # Display end time with nanoseconds
    echo "----------------------------------"
    
    # Calculate the duration using bc for floating point arithmetic
    duration=$(echo "$end_time - $start_time" | bc -l) # -l loads the math library
    
    # Output the result
    echo "=================================="
    echo " Training Monitoring Complete"
    echo "=================================="
    echo "Log File:         $LOGFILE"
    echo "Start Condition:  File created"
    echo "Stop Condition:   Found \"$DONE_STRING\""
    printf "Total Duration:   %.3f seconds\n" "$duration" # Format to 3 decimal places
    echo "=================================="
    break # Exit the monitoring loop
  fi
  # Brief sleep to prevent tight loop if inotifywait times out repeatedly without events.
  # This is mostly a fallback; inotifywait should block until an event.
  sleep 0.1
done

echo "Monitoring script finished."
