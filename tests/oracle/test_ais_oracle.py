"""The AIS messages of comnum against pyais (recorded).

pyais encodes the fields of ITU-R M.1371-5, Annex 8, into ``!AIVDO`` sentences and
decodes them back. The oracle records, for messages 1, 4, 5, 18 and 24 (both parts)
with chosen fields, the sentences pyais writes and the fields it reads from them; the
test parses the recorded sentences (``comnum.ais_messages``) and checks that
``comnum.ais_decode`` reads the same fields, and that a position report built by comnum
(``comnum.ais_position_report``, ``comnum.ais_aivdm``) is read by pyais with the fields
it was given (``tests/oracle/ais_messages.json``, see ``tests/oracle/__init__.py``):
pyais is imported only inside the recording. Re-record with

    COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_ais_oracle.py
"""

from __future__ import annotations

import numpy as np
import pytest

import comnum
from oracle import oracle

# The messages pyais encodes: the fields of its `encode_dict`.
MESSAGES = [
    dict(
        type=1,
        mmsi=200123456,
        status=0,
        turn=-128,
        speed=12.3,
        accuracy=1,
        lon=-9.8765,
        lat=46.1234,
        course=271.5,
        heading=270,
        second=42,
        maneuver=0,
        raim=0,
    ),
    dict(
        type=4,
        mmsi=2005000,
        year=2026,
        month=11,
        day=4,
        hour=19,
        minute=30,
        second=15,
        accuracy=1,
        lon=-9.88,
        lat=46.12,
        epfd=1,
        raim=0,
    ),
    dict(
        type=5,
        mmsi=200123456,
        ais_version=1,
        imo=1000000,
        callsign="TEST1",
        shipname="COMNUM TEST",
        ship_type=52,
        to_bow=50,
        to_stern=30,
        to_port=8,
        to_starboard=8,
        epfd=1,
        month=11,
        day=5,
        hour=6,
        minute=30,
        draught=6.5,
        destination="NOWHERE",
        dte=0,
    ),
    dict(
        type=18,
        mmsi=200654321,
        speed=5.1,
        accuracy=0,
        lon=-9.90,
        lat=46.15,
        course=100.0,
        heading=511,
        second=7,
        cs=1,
        raim=0,
    ),
    dict(type=24, mmsi=200654321, partno=0, shipname="TEST VESSEL"),
    dict(
        type=24,
        mmsi=200654321,
        partno=1,
        ship_type=36,
        vendorid="COM",
        callsign="TEST2",
        to_bow=5,
        to_stern=4,
        to_port=1,
        to_starboard=2,
    ),
]

# A position report built by comnum, read by pyais: (mmsi, status, lat, lon, sog, cog, heading, second).
REPORT = (200999999, 1, -33.8688, 151.2093, 102.2, 359.9, 0, 59)

FIELDS = (
    "mmsi",
    "lat",
    "lon",
    "speed",
    "course",
    "heading",
    "second",
    "status",
    "imo",
    "callsign",
    "shipname",
    "ship_type",
    "destination",
    "draught",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "partno",
    "vendorid",
    "to_bow",
    "to_stern",
    "to_port",
    "to_starboard",
    "epfd",
    "cs",
    "dte",
    "ais_version",
    "accuracy",
    "raim",
)


def _plain(v):
    """A JSON value of a field of pyais (its enumerations as integers)."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, str)) or v is None:
        return v
    return int(v)


def compute() -> dict:
    from pyais import decode
    from pyais.encode import encode_dict

    out = {}
    for k, m in enumerate(MESSAGES):
        sentences = encode_dict(m, radio_channel="A", talker_id="AI")
        d = decode(*sentences).asdict()
        out[str(k)] = {
            "sentences": sentences,
            "fields": {f: _plain(d[f]) for f in FIELDS if f in d},
        }
    mmsi, status, lat, lon, sog, cog, heading, second = REPORT
    bits = comnum.ais_position_report(
        mmsi,
        nav_status=status,
        lat_deg=lat,
        lon_deg=lon,
        sog_knots=sog,
        cog_deg=cog,
        heading_deg=heading,
        timestamp=second,
    )
    sentences = comnum.ais_aivdm(bits)
    d = decode(*sentences).asdict()
    out["report"] = {"sentences": sentences, "fields": {f: _plain(d[f]) for f in FIELDS if f in d}}
    return out


def _position_matches(p: dict, f: dict) -> None:
    for ours, key, na in (
        (p["lat_deg"], "lat", 91.0),
        (p["lon_deg"], "lon", 181.0),
        (p["sog_knots"], "speed", 102.3),
        (p["cog_deg"], "course", 360.0),
    ):
        if key not in f:
            continue
        if ours is None:
            assert f[key] == pytest.approx(na), (key, f[key])
        else:
            assert ours == pytest.approx(f[key], abs=1e-4), (key, ours, f[key])
    if "heading" in f:
        assert (511 if p["heading_deg"] is None else p["heading_deg"]) == f["heading"]
    if "second" in f:
        assert p["timestamp"] == f["second"]
    if "accuracy" in f:
        assert int(p["accurate"]) == f["accuracy"]


def test_ais_decode_matches_pyais():
    rec = oracle("ais_messages", compute, tool="pyais")
    for k in range(len(MESSAGES)):
        sentences, f = rec[str(k)]["sentences"], rec[str(k)]["fields"]
        parsed = comnum.ais_messages(list(sentences))
        assert len(parsed) == 1 and parsed[0]["channel"] == "A"
        m = comnum.ais_decode(parsed[0]["bits"])
        assert (m["msg_type"], m["mmsi"]) == (MESSAGES[k]["type"], f["mmsi"])
        if m["msg_type"] in (1, 18):
            _position_matches(m["position"], f)
            if m["msg_type"] == 1:
                assert (m["nav_status"], m["raim"]) == (f["status"], bool(f["raim"]))
            else:
                assert m["cs_unit"] == bool(f["cs"])
        elif m["msg_type"] == 4:
            assert m["utc"] == (
                f["year"],
                f["month"],
                f["day"],
                f["hour"],
                f["minute"],
                f["second"],
            )
            _position_matches(
                m["position"], {k2: v for k2, v in f.items() if k2 in ("lat", "lon", "accuracy")}
            )
            assert m["epfd"] == f["epfd"]
        elif m["msg_type"] == 5:
            assert (m["imo"], m["callsign"], m["shipname"], m["ship_type"]) == (
                f["imo"],
                f["callsign"],
                f["shipname"],
                f["ship_type"],
            )
            assert (m["destination"], m["draught_m"], m["dte"], m["ais_version"], m["epfd"]) == (
                f["destination"],
                f["draught"],
                bool(f["dte"]),
                f["ais_version"],
                f["epfd"],
            )
            assert m["eta"] == (f["month"], f["day"], f["hour"], f["minute"])
            assert m["dimensions"] == {
                "to_bow": f["to_bow"],
                "to_stern": f["to_stern"],
                "to_port": f["to_port"],
                "to_starboard": f["to_starboard"],
            }
        elif m["msg_type"] == 24:
            assert m["part"] == f["partno"]
            if m["part"] == 0:
                assert m["shipname"] == f["shipname"]
            else:
                assert (m["ship_type"], m["vendor"], m["callsign"]) == (
                    f["ship_type"],
                    f["vendorid"],
                    f["callsign"],
                )
                assert m["dimensions"] == {
                    "to_bow": f["to_bow"],
                    "to_stern": f["to_stern"],
                    "to_port": f["to_port"],
                    "to_starboard": f["to_starboard"],
                }
    # the report of comnum as pyais reads it
    mmsi, status, lat, lon, sog, cog, heading, second = REPORT
    f = rec["report"]["fields"]
    assert (f["mmsi"], f["status"], f["heading"], f["second"]) == (mmsi, status, heading, second)
    assert (f["lat"], f["lon"], f["speed"], f["course"]) == pytest.approx(
        (lat, lon, sog, cog), abs=1e-4
    )
    assert isinstance(comnum.ais_messages(list(rec["report"]["sentences"]))[0]["bits"], np.ndarray)
