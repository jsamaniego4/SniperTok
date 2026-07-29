from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random

import numpy as np
import pandas as pd


CATEGORIES = {
    "beauty": [
        "LED Face Mask", "Heatless Curl Set", "Glow Serum", "Ice Roller", "Lip Plumper"
    ],
    "home": [
        "Mini Mop", "Cloud Humidifier", "Shelf Organizer", "Sink Caddy", "Cordless Spin Scrubber"
    ],
    "fitness": [
        "Ab Roller", "Resistance Bands", "Walking Pad", "Massage Gun", "Grip Trainer"
    ],
    "pets": [
        "Slow Feeder Bowl", "Pet Hair Remover", "Cat Water Fountain", "Dog Seat Cover", "Treat Puzzle"
    ],
    "car": [
        "Mini Car Vacuum", "Interior Detail Gel", "Phone Mount", "Wheel Brush", "Seat Gap Filler"
    ],
    "kitchen": [
        "Mini Chopper", "Oil Sprayer", "Meal Prep Box", "Silicone Air Fryer Liner", "Portable Blender"
    ],
    "tech": [
        "MagSafe Charger", "RGB Light Bar", "Cable Organizer", "Portable Monitor", "Desk Dock"
    ],
}

PLATFORM = "tiktok"


def _choose_product():
    category = random.choice(list(CATEGORIES.keys()))
    product = random.choice(CATEGORIES[category])
    return category, product


def _text_for(category: str, product: str) -> tuple[str, str]:
    hooks = {
        "beauty": ["amazon beauty find", "viral skincare pick", "glow up favorite"],
        "home": ["clean tok made me buy it", "home refresh essential", "small space hack"],
        "fitness": ["gym bag essential", "fitness tiktok made me buy it", "recovery favorite"],
        "pets": ["pet parent must have", "cat tok favorite", "dog owner hack"],
        "car": ["car tok favorite", "must have car accessory", "detailing hack"],
        "kitchen": ["kitchen gadget worth it", "meal prep favorite", "amazon home find"],
        "tech": ["desk setup essential", "tech under 50", "amazon gadget favorite"],
    }
    hashtags = {
        "beauty": "#beauty #skincare #amazonfinds #tiktokmademebuyit",
        "home": "#home #cleantok #amazonhome #organizing",
        "fitness": "#gym #fitness #recovery #amazonfinds",
        "pets": "#pets #cattok #dogtok #petmusthaves",
        "car": "#car #cartok #detailing #amazoncar",
        "kitchen": "#kitchen #mealprep #amazonfinds #homegadgets",
        "tech": "#tech #desksetup #gadgets #amazonfinds",
    }
    caption = f"{random.choice(hooks[category])} {product.lower()} and I get why everyone is buying it"
    return caption, hashtags[category]


def generate_dataset(rows: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    base_time = datetime.now(timezone.utc) - timedelta(days=30)
    records = []

    creator_pool = [f"creator_{i:04d}" for i in range(800)]

    for idx in range(rows):
        category, product = _choose_product()
        caption, hashtags = _text_for(category, product)

        created_at = base_time + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        creator_id = random.choice(creator_pool)
        creator_followers = int(np.random.lognormal(mean=10.2, sigma=0.75))

        virality_boost = {
            "beauty": 1.10,
            "home": 1.00,
            "fitness": 0.95,
            "pets": 1.05,
            "car": 0.98,
            "kitchen": 1.02,
            "tech": 1.08,
        }[category]

        views = int(np.random.lognormal(mean=10.4, sigma=0.9) * virality_boost)
        likes = int(views * np.random.uniform(0.03, 0.12))
        comments = int(views * np.random.uniform(0.001, 0.012))
        shares = int(views * np.random.uniform(0.002, 0.02))
        saves = int(views * np.random.uniform(0.003, 0.025))

        video_length_sec = float(np.random.uniform(10, 45))
        retention_bias = {
            "beauty": 0.72,
            "home": 0.69,
            "fitness": 0.63,
            "pets": 0.74,
            "car": 0.66,
            "kitchen": 0.70,
            "tech": 0.68,
        }[category]
        watch_time_avg = round(video_length_sec * np.random.uniform(retention_bias * 0.7, retention_bias * 1.15), 2)

        records.append({
            "post_id": f"post_{idx:06d}",
            "created_at": created_at.isoformat(sep=" "),
            "platform": PLATFORM,
            "creator_id": creator_id,
            "creator_followers": creator_followers,
            "caption": caption,
            "hashtags": hashtags,
            "product_name": product,
            "product_category": category,
            "views": views,
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "saves": saves,
            "watch_time_avg": watch_time_avg,
            "video_length_sec": round(video_length_sec, 2),
        })

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SniperTok data")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/sample/sample_posts.csv")
    args = parser.parse_args()

    df = generate_dataset(rows=args.rows, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
