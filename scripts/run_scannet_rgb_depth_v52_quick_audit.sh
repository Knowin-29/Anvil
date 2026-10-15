#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -r "$ROOT/.env" ]] && source "$ROOT/.env"
PYTHON_REQUEST="${PYTHON_BIN:-${GSA_PYTHON_BIN:-python}}"
PYTHON_BIN="$(command -v "$PYTHON_REQUEST" 2>/dev/null || true)"
PACK="${PACK:-$ROOT/data/semantic_audit_v52_scannet_unique_referent_gs_formal}"
SERVER="$ROOT/tools/quick_scannet_rgb_depth_v52_unique_referent_audit.py"
SERVER_C="$ROOT/tools/quick_scannet_rgb_depth_v52_adjudication.py"
LOG_DIR="${LOG_DIR:-$ROOT/artifacts/audits/v52_quick_audit_live}"
PORT_A="${PORT_A:-10323}"
PORT_B="${PORT_B:-10324}"
PORT_C="${PORT_C:-10325}"
WATCH_INTERVAL_SECONDS="${WATCH_INTERVAL_SECONDS:-5}"
HOST="${HOST:-127.0.0.1}"
TOKEN="${TOKEN:-${GSA_QUICK_AUDIT_TOKEN:-}}"
HTTP_HOST="$HOST"
[[ "$HOST" == "0.0.0.0" ]] && HTTP_HOST=127.0.0.1
ACTION="${1:-start}"

if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && "$HOST" != "::1" \
   && -z "$TOKEN" ]]; then
  echo "A non-loopback HOST requires TOKEN or GSA_QUICK_AUDIT_TOKEN" >&2
  exit 2
fi
export GSA_QUICK_AUDIT_TOKEN="$TOKEN"
CURL_AUTH=()
[[ -z "$TOKEN" ]] || CURL_AUTH=(--header "Authorization: Bearer $TOKEN")

command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 2; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ "$PORT_A" =~ ^[0-9]+$ && "$PORT_B" =~ ^[0-9]+$ && "$PORT_C" =~ ^[0-9]+$ \
   && "$PORT_A" != "$PORT_B" && "$PORT_A" != "$PORT_C" && "$PORT_B" != "$PORT_C" ]] || {
  echo "PORT_A, PORT_B, and PORT_C must be distinct numeric ports" >&2
  exit 2
}
[[ "$WATCH_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "WATCH_INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
}
if [[ -n "$TOKEN" ]]; then
  tmux set-environment -g GSA_QUICK_AUDIT_TOKEN "$TOKEN"
else
  tmux set-environment -gu GSA_QUICK_AUDIT_TOKEN 2>/dev/null || true
fi

session_name() {
  local side="$1" port="$2"
  printf 'gsa_v52_%s_%s' "$side" "$port"
}

watcher_session_name() {
  printf 'gsa_v52_watcher_%s_%s_%s' "$PORT_A" "$PORT_B" "$PORT_C"
}

watcher_log_path() {
  printf '%s/reviewer_c_watcher_%s.log' "$LOG_DIR" "$PORT_C"
}

watcher_state_path() {
  printf '%s/reviewer_c_watcher_%s.state.tsv' "$LOG_DIR" "$PORT_C"
}

write_watcher_state() {
  local state="$1" detail="$2" path temporary
  path="$(watcher_state_path)"
  mkdir -p "$LOG_DIR"
  temporary="${path}.$$"
  printf 'timestamp_utc\tstate\tdetail\n%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" "${detail//$'\t'/ }" >"$temporary"
  mv -f "$temporary" "$path"
}

log_path() {
  local side="$1" port="$2"
  if [[ "$side" == c ]]; then
    printf '%s/reviewer_c_%s.log' "$LOG_DIR" "$port"
    return
  fi
  printf '%s/annotator_%s_%s.log' "$LOG_DIR" "$side" "$port"
}

preflight_side() {
  local side="$1" port="$2" output
  output="$($PYTHON_BIN "$SERVER" --pack "$PACK" --annotator "$side" --host "$HOST" --port "$port" --check)"
  "$PYTHON_BIN" -c '
import json, sys
row = json.load(sys.stdin)
assert row["images"] == 33, row
assert row["expected_decisions"] == 33, row
assert 0 <= row["decisions"] <= 33, row
assert row["complete_images"] == row["decisions"], row
' <<<"$output"
  printf 'annotator_%s preflight: %s\n' "$side" "$(tr -d '\n ' <<<"$output")"
}

c_state_json() {
  "$PYTHON_BIN" "$SERVER_C" --pack "$PACK" --host "$HOST" --port "$PORT_C" --check
}

c_field() {
  local field="$1"
  "$PYTHON_BIN" -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$field"
}

preflight_c() {
  local output state
  output="$(c_state_json)"
  state="$(c_field state <<<"$output")"
  printf 'reviewer_c preflight: state=%s stats=%s\n' "$state" "$(tr -d '\n ' <<<"$output")"
}

listener() {
  local port="$1"
  ss -H -ltnp "sport = :$port" 2>/dev/null || true
}

assert_managed_listener() {
  local port="$1" line
  line="$(listener "$port")"
  [[ -n "$line" ]] || { echo "port $port is not listening" >&2; return 1; }
  case "$HOST" in
    127.0.0.1|localhost)
      grep -Fq "127.0.0.1:$port" <<<"$line" || {
        echo "port $port is not bound to loopback: $line" >&2; return 1; }
      ;;
    0.0.0.0)
      grep -Fq "0.0.0.0:$port" <<<"$line" || {
        echo "port $port is not bound to all IPv4 interfaces: $line" >&2; return 1; }
      ;;
    *)
      grep -Fq "$HOST:$port" <<<"$line" || {
        echo "port $port is not bound to $HOST: $line" >&2; return 1; }
      ;;
  esac
}

http_check() {
  local side="$1" port="$2" path code
  for path in / /api/stats /api/case/0 /api/image/0; do
    code="$(curl --noproxy '*' --silent --show-error --output /dev/null --max-time 8 \
      "${CURL_AUTH[@]}" --write-out '%{http_code}' "http://$HTTP_HOST:$port$path")"
    [[ "$code" == 200 ]] || {
      echo "annotator_$side $path returned HTTP $code" >&2
      return 1
    }
  done
  printf 'annotator_%s HTTP: page/stats/case/image = 200/200/200/200\n' "$side"
}

start_side() {
  local side="$1" port="$2" session log command_line
  session="$(session_name "$side" "$port")"
  log="$(log_path "$side" "$port")"
  if tmux has-session -t "$session" 2>/dev/null; then
    assert_managed_listener "$port"
    http_check "$side" "$port"
    echo "annotator_$side already running in tmux session $session"
    return
  fi
  if [[ -n "$(listener "$port")" ]]; then
    echo "refusing to replace an unmanaged listener on port $port: $(listener "$port")" >&2
    return 3
  fi
  mkdir -p "$LOG_DIR"
  printf -v command_line '%q ' env PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "$SERVER" \
    --pack "$PACK" --annotator "$side" --host "$HOST" --port "$port"
  printf -v command_line '%s >> %q 2>&1' "$command_line" "$log"
  tmux new-session -d -s "$session" "$command_line"
  for _ in {1..40}; do
    if assert_managed_listener "$port" >/dev/null 2>&1 && \
       curl --noproxy '*' --silent --fail --output /dev/null --max-time 2 \
         "${CURL_AUTH[@]}" "http://$HTTP_HOST:$port/api/stats"; then
      http_check "$side" "$port"
      echo "annotator_$side started: session=$session log=$log"
      return
    fi
    sleep 0.25
  done
  echo "annotator_$side failed to start; inspect $log" >&2
  tail -n 40 "$log" >&2 || true
  exit 4
}

start_c() {
  local side=c port="$PORT_C" session log command_line
  session="$(session_name "$side" "$port")"
  log="$(log_path "$side" "$port")"
  if tmux has-session -t "$session" 2>/dev/null; then
    assert_managed_listener "$port"
    http_check "$side" "$port"
    echo "reviewer_c already running in tmux session $session"
    return
  fi
  if [[ -n "$(listener "$port")" ]]; then
    echo "refusing to replace an unmanaged listener on port $port: $(listener "$port")" >&2
    return 3
  fi
  mkdir -p "$LOG_DIR"
  [[ -z "$TOKEN" ]] || tmux set-environment -g GSA_QUICK_AUDIT_TOKEN "$TOKEN"
  printf -v command_line '%q ' env PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" "$SERVER_C" \
    --pack "$PACK" --host "$HOST" --port "$port"
  printf -v command_line '%s >> %q 2>&1' "$command_line" "$log"
  tmux new-session -d -s "$session" "$command_line"
  for _ in {1..40}; do
    if assert_managed_listener "$port" >/dev/null 2>&1 && \
       curl --noproxy '*' --silent --fail --output /dev/null --max-time 2 \
         "${CURL_AUTH[@]}" "http://$HTTP_HOST:$port/api/stats"; then
      http_check "$side" "$port"
      echo "reviewer_c started: session=$session log=$log"
      return
    fi
    sleep 0.25
  done
  echo "reviewer_c failed to start; inspect $log" >&2
  tail -n 40 "$log" >&2 || true
  return 4
}

stop_side() {
  local side="$1" port="$2" session
  session="$(session_name "$side" "$port")"
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session"
    echo "stopped $session"
  else
    echo "$session is not running"
  fi
}

status_side() {
  local side="$1" port="$2" session stats listener_line listener_pid pane_pid
  session="$(session_name "$side" "$port")"
  tmux has-session -t "$session" 2>/dev/null || {
    echo "annotator_$side: STOPPED session=$session"
    return 1
  }
  assert_managed_listener "$port"
  http_check "$side" "$port"
  stats="$(curl --noproxy '*' --silent --fail --max-time 5 \
    "${CURL_AUTH[@]}" "http://$HTTP_HOST:$port/api/stats")"
  listener_line="$(listener "$port")"
  listener_pid="$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' <<<"$listener_line")"
  pane_pid="$(tmux display-message -p -t "$session" '#{pane_pid}')"
  printf 'annotator_%s: RUNNING pid=%s pane_pid=%s session=%s url=http://%s:%s/ stats=%s log=%s\n' \
    "$side" "$listener_pid" "$pane_pid" "$session" "$HOST" "$port" "$stats" "$(log_path "$side" "$port")"
}

ab_services_healthy() {
  local side port session
  for side in a b; do
    if [[ "$side" == a ]]; then port="$PORT_A"; else port="$PORT_B"; fi
    session="$(session_name "$side" "$port")"
    tmux has-session -t "$session" 2>/dev/null || return 1
    assert_managed_listener "$port" >/dev/null 2>&1 || return 1
    curl --noproxy '*' --silent --fail --output /dev/null --max-time 3 \
      "${CURL_AUTH[@]}" "http://$HTTP_HOST:$port/api/stats" || return 1
  done
}

c_service_healthy() {
  local session
  session="$(session_name c "$PORT_C")"
  tmux has-session -t "$session" 2>/dev/null || return 1
  assert_managed_listener "$PORT_C" >/dev/null 2>&1 || return 1
  curl --noproxy '*' --silent --fail --output /dev/null --max-time 3 \
    "${CURL_AUTH[@]}" "http://$HTTP_HOST:$PORT_C/api/stats" || return 1
}

sync_c_service() {
  local output state session detail
  session="$(session_name c "$PORT_C")"
  if ! ab_services_healthy; then
    tmux has-session -t "$session" 2>/dev/null && stop_side c "$PORT_C"
    write_watcher_state ERROR_A_B_SERVICE_UNHEALTHY \
      "watcher retained; start/restart the missing A/B service"
    return 1
  fi
  if ! output="$(c_state_json 2>&1)"; then
    tmux has-session -t "$session" 2>/dev/null && stop_side c "$PORT_C"
    write_watcher_state ERROR_C_STATE "$(tr '\n\t' '  ' <<<"$output")"
    return 1
  fi
  if ! state="$(c_field state <<<"$output" 2>/dev/null)"; then
    tmux has-session -t "$session" 2>/dev/null && stop_side c "$PORT_C"
    write_watcher_state ERROR_C_STATE "invalid adjudication-state JSON"
    return 1
  fi
  detail="$(tr -d '\n' <<<"$output")"
  case "$state" in
    READY_FOR_ADJUDICATION)
      if ! c_service_healthy; then
        tmux has-session -t "$session" 2>/dev/null && stop_side c "$PORT_C"
        if [[ -n "$(listener "$PORT_C")" ]]; then
          write_watcher_state ERROR_C_START \
            "unmanaged listener blocks port $PORT_C: $(listener "$PORT_C")"
          return 1
        fi
        if ! start_c; then
          write_watcher_state ERROR_C_START "reviewer C failed to start; see $(log_path c "$PORT_C")"
          return 1
        fi
      fi
      write_watcher_state READY_C_RUNNING "$detail"
      ;;
    WAITING_FOR_A_B|NO_ADJUDICATION_REQUIRED)
      if tmux has-session -t "$session" 2>/dev/null; then
        stop_side c "$PORT_C"
      fi
      if [[ -n "$(listener "$PORT_C")" ]]; then
        write_watcher_state ERROR_C_START \
          "reviewer C must not listen in state $state: $(listener "$PORT_C")"
        return 1
      fi
      write_watcher_state "$state" "$detail"
      ;;
    *)
      write_watcher_state ERROR_C_STATE "unexpected reviewer C state: $state"
      return 1
      ;;
  esac
}

watcher_loop() {
  trap 'write_watcher_state STOPPED "watcher received signal"; exit 0' HUP INT TERM
  write_watcher_state STARTING "poll_interval_seconds=$WATCH_INTERVAL_SECONDS"
  while true; do
    sync_c_service || true
    sleep "$WATCH_INTERVAL_SECONDS" &
    wait $! || true
  done
}

status_watcher() {
  local session state_path state_line pane_pid
  session="$(watcher_session_name)"
  state_path="$(watcher_state_path)"
  state_line="missing"
  [[ ! -s "$state_path" ]] || state_line="$(tail -n 1 "$state_path")"
  if ! tmux has-session -t "$session" 2>/dev/null; then
    printf 'reviewer_c_watcher: STOPPED session=%s last_state=%s log=%s\n' \
      "$session" "$state_line" "$(watcher_log_path)"
    return 1
  fi
  pane_pid="$(tmux display-message -p -t "$session" '#{pane_pid}')"
  printf 'reviewer_c_watcher: RUNNING pane_pid=%s session=%s state=%s log=%s\n' \
    "$pane_pid" "$session" "$state_line" "$(watcher_log_path)"
}

start_watcher() {
  local session log command_line state_path
  session="$(watcher_session_name)"
  log="$(watcher_log_path)"
  state_path="$(watcher_state_path)"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "reviewer_c_watcher already running in tmux session $session"
    status_watcher
    return
  fi
  mkdir -p "$LOG_DIR"
  write_watcher_state STARTING "launcher is creating watcher tmux"
  printf -v command_line '%q ' env \
    PYTHON_BIN="$PYTHON_BIN" PACK="$PACK" LOG_DIR="$LOG_DIR" \
    HOST="$HOST" \
    PORT_A="$PORT_A" PORT_B="$PORT_B" PORT_C="$PORT_C" \
    WATCH_INTERVAL_SECONDS="$WATCH_INTERVAL_SECONDS" \
    bash "$ROOT/scripts/run_scannet_rgb_depth_v52_quick_audit.sh" _watch
  printf -v command_line '%s >> %q 2>&1' "$command_line" "$log"
  tmux new-session -d -s "$session" "$command_line"
  for _ in {1..80}; do
    if tmux has-session -t "$session" 2>/dev/null && [[ -s "$state_path" ]] && \
       ! tail -n 1 "$state_path" | grep -q $'\tSTARTING\t'; then
      echo "reviewer_c_watcher started: session=$session log=$log"
      status_watcher
      return
    fi
    sleep 0.25
  done
  echo "reviewer_c_watcher failed to become ready; inspect $log" >&2
  tail -n 40 "$log" >&2 || true
  return 4
}

stop_watcher() {
  local session
  session="$(watcher_session_name)"
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session"
    write_watcher_state STOPPED "stopped by launcher"
    echo "stopped $session"
  else
    write_watcher_state STOPPED "watcher was not running"
    echo "$session is not running"
  fi
}

status_c() {
  local output state session
  session="$(session_name c "$PORT_C")"
  if ! output="$(c_state_json 2>&1)"; then
    printf 'reviewer_c: FAIL_CLOSED state=ERROR_C_STATE detail=%s\n' \
      "$(tr '\n\t' '  ' <<<"$output")" >&2
    return 1
  fi
  if ! state="$(c_field state <<<"$output" 2>/dev/null)"; then
    echo "reviewer_c: FAIL_CLOSED state=ERROR_C_STATE detail=invalid_state_json" >&2
    return 1
  fi
  case "$state" in
    READY_FOR_ADJUDICATION)
      status_side c "$PORT_C"
      ;;
    WAITING_FOR_A_B|NO_ADJUDICATION_REQUIRED)
      if tmux has-session -t "$session" 2>/dev/null || [[ -n "$(listener "$PORT_C")" ]]; then
        echo "reviewer_c unexpectedly active in state $state" >&2
        return 1
      fi
      printf 'reviewer_c: %s service=NOT_STARTED stats=%s\n' "$state" "$(tr -d '\n ' <<<"$output")"
      ;;
    *)
      echo "unexpected reviewer C state: $state" >&2
      return 1
      ;;
  esac
}

case "$ACTION" in
  _watch)
    watcher_loop
    ;;
  check)
    preflight_side a "$PORT_A"
    preflight_side b "$PORT_B"
    preflight_c
    ;;
  start)
    preflight_side a "$PORT_A"
    preflight_side b "$PORT_B"
    start_side a "$PORT_A"
    start_side b "$PORT_B"
    status_side a "$PORT_A"
    status_side b "$PORT_B"
    start_watcher
    status_c
    ;;
  status)
    failed=0
    status_side a "$PORT_A" || failed=1
    status_side b "$PORT_B" || failed=1
    status_watcher || failed=1
    status_c || failed=1
    exit "$failed"
    ;;
  stop)
    stop_watcher
    stop_side c "$PORT_C"
    stop_side a "$PORT_A"
    stop_side b "$PORT_B"
    ;;
  restart)
    stop_watcher
    stop_side c "$PORT_C"
    stop_side a "$PORT_A"
    stop_side b "$PORT_B"
    preflight_side a "$PORT_A"
    preflight_side b "$PORT_B"
    start_side a "$PORT_A"
    start_side b "$PORT_B"
    status_side a "$PORT_A"
    status_side b "$PORT_B"
    start_watcher
    status_c
    ;;
  *)
    echo "usage: $0 {start|status|check|stop|restart}" >&2
    exit 2
    ;;
esac
