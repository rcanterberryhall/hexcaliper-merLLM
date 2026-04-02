"""
geoip.py — IP-to-UTC-offset lookup using MaxMind GeoLite2-City.

If the database is absent or the lookup fails, returns 0 (UTC).
The resolved offset is cached; updates when the detected offset changes.

Download GeoLite2-City.mmdb from MaxMind (free account required) and place
it at the path specified by GEOIP_DB_PATH.
"""
import os
from typing import Optional

import config

_reader = None
_cached_ip: Optional[str] = None
_cached_offset: float = 0.0


def _get_reader():
    global _reader
    if _reader is not None:
        return _reader
    if not os.path.exists(config.GEOIP_DB_PATH):
        return None
    try:
        import geoip2.database
        _reader = geoip2.database.Reader(config.GEOIP_DB_PATH)
        return _reader
    except Exception:
        return None


def get_utc_offset(ip: str) -> float:
    """
    Return the UTC offset in hours for the given IP address.

    Returns the manual override if configured, otherwise performs a GeoIP
    lookup. Falls back to 0.0 (UTC) on any error.

    :param ip: Client IP address string.
    :return: UTC offset in hours (e.g. -5.0 for EST).
    """
    global _cached_ip, _cached_offset

    if config.GEOIP_OFFSET_OVERRIDE:
        try:
            return float(config.GEOIP_OFFSET_OVERRIDE)
        except ValueError:
            pass

    if ip == _cached_ip:
        return _cached_offset

    reader = _get_reader()
    if not reader:
        return 0.0

    try:
        response = reader.city(ip)
        offset = float(response.location.time_zone and
                       _tz_to_offset(response.location.time_zone) or 0.0)
    except Exception:
        offset = 0.0

    _cached_ip = ip
    _cached_offset = offset
    return offset


def _tz_to_offset(tz_name: str) -> float:
    """Convert IANA timezone name to current UTC offset in hours."""
    try:
        from datetime import datetime
        import zoneinfo
        now = datetime.now(zoneinfo.ZoneInfo(tz_name))
        return now.utcoffset().total_seconds() / 3600
    except Exception:
        return 0.0


def day_end_utc(local_day_end: str, utc_offset: float) -> tuple[int, int]:
    """
    Convert a local HH:MM day-end time to UTC hours/minutes.

    :param local_day_end: Local time string "HH:MM".
    :param utc_offset: UTC offset in hours.
    :return: (hour, minute) in UTC.
    """
    try:
        h, m = map(int, local_day_end.split(":"))
    except ValueError:
        h, m = 22, 0

    total_minutes = h * 60 + m - int(utc_offset * 60)
    total_minutes %= 1440
    return divmod(total_minutes, 60)
