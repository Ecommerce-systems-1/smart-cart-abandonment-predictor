"""
Synthetic session generator for Smart Cart Abandonment Predictor.

Usage:
    python data/generate.py --scenario happy_path --volume 1000 --seed 42
    python data/generate.py --scenario high_risk --volume 500 --seed 42
    python data/generate.py --scenario edge_cases --volume 200 --seed 42
    python data/generate.py --scenario mixed --volume 10000 --seed 42

Scenarios:
    happy_path   Low abandonment rate (~25%); desktop-heavy; high cart values
    high_risk    High abandonment rate (~85%); mobile-heavy; low cart values; many removes
    edge_cases   Zero cart value, single-item carts, very long sessions, repeated checkout attempts
    mixed        Realistic distribution (~70% abandonment); balanced device/referrer mix
"""

import argparse
import csv
import random
import sys
import uuid
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np


DEVICE_TYPES = ["mobile", "desktop", "tablet"]
REFERRER_SOURCES = ["organic", "paid", "email", "direct", "social"]

Scenario = Literal["happy_path", "high_risk", "edge_cases", "mixed"]


@dataclass
class SessionRecord:
    session_id: str
    cart_value: float
    item_count: int
    cart_adds: int
    cart_removes: int
    time_since_last_event_seconds: float
    session_duration_seconds: float
    page_views: int
    checkout_start_attempts: int
    device_type: str
    referrer_source: str
    time_on_checkout_seconds: float
    abandoned: int  # label: 1 = abandoned, 0 = converted


def _generate_session(scenario: Scenario, rng: random.Random, np_rng: np.random.Generator) -> SessionRecord:
    if scenario == "happy_path":
        device = rng.choices(DEVICE_TYPES, weights=[0.2, 0.7, 0.1])[0]
        cart_value = float(np_rng.normal(120, 40).clip(20, 500))
        item_count = rng.randint(2, 8)
        cart_adds = item_count + rng.randint(0, 2)
        cart_removes = rng.randint(0, 1)
        page_views = rng.randint(5, 20)
        checkout_attempts = rng.randint(1, 2)
        time_on_checkout = float(np_rng.normal(90, 30).clip(10, 300))
        session_duration = float(np_rng.normal(600, 120).clip(60, 1800))
        time_since_last = float(np_rng.exponential(15).clip(1, 60))
        abandoned = 1 if rng.random() < 0.25 else 0

    elif scenario == "high_risk":
        device = rng.choices(DEVICE_TYPES, weights=[0.75, 0.2, 0.05])[0]
        cart_value = float(np_rng.normal(35, 15).clip(5, 120))
        item_count = rng.randint(1, 3)
        cart_adds = item_count + rng.randint(1, 4)
        cart_removes = rng.randint(1, cart_adds)
        page_views = rng.randint(2, 8)
        checkout_attempts = rng.randint(0, 3)
        time_on_checkout = float(np_rng.normal(20, 15).clip(0, 90))
        session_duration = float(np_rng.normal(200, 80).clip(30, 600))
        time_since_last = float(np_rng.exponential(120).clip(30, 600))
        abandoned = 1 if rng.random() < 0.85 else 0

    elif scenario == "edge_cases":
        edge = rng.choice(["zero_cart", "single_item", "long_session", "repeated_checkout"])
        device = rng.choice(DEVICE_TYPES)
        if edge == "zero_cart":
            cart_value = 0.0
            item_count = 0
            cart_adds = rng.randint(0, 1)
            cart_removes = cart_adds
            page_views = rng.randint(1, 3)
            checkout_attempts = 0
            time_on_checkout = 0.0
            session_duration = float(np_rng.exponential(60).clip(5, 300))
            time_since_last = float(np_rng.exponential(30).clip(5, 300))
            abandoned = 1
        elif edge == "single_item":
            cart_value = float(np_rng.normal(25, 10).clip(1, 100))
            item_count = 1
            cart_adds = 1
            cart_removes = 0
            page_views = rng.randint(1, 5)
            checkout_attempts = rng.randint(0, 1)
            time_on_checkout = float(np_rng.normal(30, 20).clip(0, 120))
            session_duration = float(np_rng.normal(180, 60).clip(30, 600))
            time_since_last = float(np_rng.exponential(60).clip(5, 300))
            abandoned = 1 if rng.random() < 0.65 else 0
        elif edge == "long_session":
            cart_value = float(np_rng.normal(200, 60).clip(50, 800))
            item_count = rng.randint(5, 15)
            cart_adds = item_count + rng.randint(3, 8)
            cart_removes = rng.randint(2, 5)
            page_views = rng.randint(20, 60)
            checkout_attempts = rng.randint(2, 5)
            time_on_checkout = float(np_rng.normal(300, 100).clip(60, 900))
            session_duration = float(np_rng.normal(3600, 600).clip(1800, 7200))
            time_since_last = float(np_rng.exponential(5).clip(1, 30))
            abandoned = 1 if rng.random() < 0.35 else 0
        else:  # repeated_checkout
            cart_value = float(np_rng.normal(80, 30).clip(20, 300))
            item_count = rng.randint(2, 5)
            cart_adds = item_count
            cart_removes = 0
            page_views = rng.randint(8, 20)
            checkout_attempts = rng.randint(3, 8)
            time_on_checkout = float(np_rng.normal(400, 100).clip(120, 900))
            session_duration = float(np_rng.normal(900, 300).clip(300, 2400))
            time_since_last = float(np_rng.exponential(10).clip(1, 60))
            abandoned = 1 if rng.random() < 0.50 else 0

    else:  # mixed — realistic distribution
        device = rng.choices(DEVICE_TYPES, weights=[0.55, 0.38, 0.07])[0]
        cart_value = float(np_rng.lognormal(4.0, 0.8).clip(5, 1000))
        item_count = rng.randint(1, 10)
        cart_adds = item_count + rng.randint(0, 3)
        cart_removes = rng.randint(0, min(cart_adds, 3))
        page_views = rng.randint(2, 25)
        checkout_attempts = rng.randint(0, 3)
        time_on_checkout = float(np_rng.exponential(60).clip(0, 600))
        session_duration = float(np_rng.lognormal(5.5, 0.9).clip(30, 3600))
        time_since_last = float(np_rng.exponential(45).clip(1, 600))
        abandon_prob = 0.70
        if device == "mobile":
            abandon_prob += 0.10
        if cart_value < 30:
            abandon_prob += 0.10
        if cart_removes > cart_adds * 0.3:
            abandon_prob += 0.08
        if checkout_attempts >= 2:
            abandon_prob -= 0.15
        abandoned = 1 if rng.random() < min(abandon_prob, 0.95) else 0

    referrer = rng.choices(
        REFERRER_SOURCES,
        weights=[0.30, 0.25, 0.15, 0.20, 0.10],
    )[0]

    return SessionRecord(
        session_id=str(uuid.uuid4()),
        cart_value=round(cart_value, 2),
        item_count=item_count,
        cart_adds=cart_adds,
        cart_removes=cart_removes,
        time_since_last_event_seconds=round(time_since_last, 1),
        session_duration_seconds=round(session_duration, 1),
        page_views=page_views,
        checkout_start_attempts=checkout_attempts,
        device_type=device,
        referrer_source=referrer,
        time_on_checkout_seconds=round(time_on_checkout, 1),
        abandoned=abandoned,
    )


def generate(scenario: Scenario, volume: int, seed: int) -> list[SessionRecord]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    return [_generate_session(scenario, rng, np_rng) for _ in range(volume)]


def write_csv(records: list[SessionRecord], path: str) -> None:
    fieldnames = [f.name for f in fields(SessionRecord)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic cart abandonment session data")
    parser.add_argument("--scenario", choices=["happy_path", "high_risk", "edge_cases", "mixed"], default="mixed")
    parser.add_argument("--volume", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="-", help="Output CSV path or '-' for stdout")
    args = parser.parse_args()

    records = generate(args.scenario, args.volume, args.seed)

    abandonment_rate = sum(r.abandoned for r in records) / len(records)
    print(f"Generated {len(records)} sessions | scenario={args.scenario} | abandonment_rate={abandonment_rate:.1%}", file=sys.stderr)

    if args.output == "-":
        fieldnames = [f.name for f in fields(SessionRecord)]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)
    else:
        write_csv(records, args.output)
        print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
