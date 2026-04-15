#!/usr/bin/env bash
# backup_db.sh — Safe SQLite backup for merLLM.
#
# Uses sqlite3 .backup which is safe in WAL mode while the service is running.
# Keeps the 7 most recent daily backups; older files are deleted automatically.
#
# Usage (run from any directory):
#   DB_PATH=/data/merllm.db BACKUP_DIR=/data/backups ./scripts/backup_db.sh
#
# Recommended cron (daily at 02:30):
#   30 2 * * * DB_PATH=/data/merllm.db BACKUP_DIR=/data/backups /path/to/scripts/backup_db.sh >> /var/log/merllm-backup.log 2>&1
#
# This script can also be triggered via the API:
#   POST /api/merllm/backup

set -euo pipefail

DB_PATH="${DB_PATH:-/data/merllm.db}"
BACKUP_DIR="${BACKUP_DIR:-/data/backups}"
KEEP_DAYS="${KEEP_DAYS:-7}"
APP_NAME="merllm"

if [ ! -f "$DB_PATH" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: database not found: $DB_PATH" >&2
    exit 1
fi

mkdir -p "$BACKUP_DIR"

STAMP=$(date -u +%Y%m%d-%H%M%S)
DEST="$BACKUP_DIR/${APP_NAME}-${STAMP}.db"

if sqlite3 "$DB_PATH" ".backup '$DEST'"; then
    SIZE=$(du -sh "$DEST" | cut -f1)
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] OK backup written: $DEST ($SIZE)"
else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: sqlite3 .backup failed" >&2
    rm -f "$DEST"
    exit 1
fi

# Rotate: keep only the $KEEP_DAYS most recent backups.
mapfile -t old_files < <(
    ls -1t "$BACKUP_DIR/${APP_NAME}-"*.db 2>/dev/null | tail -n +"$((KEEP_DAYS + 1))"
)
for f in "${old_files[@]}"; do
    rm -f "$f"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ROTATED $f"
done
