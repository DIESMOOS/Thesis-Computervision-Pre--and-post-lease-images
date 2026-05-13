def map_source_label(dataset_name: str, raw_label: str) -> str:
    dataset_name = dataset_name.lower().strip()
    raw_label = raw_label.lower().strip()

    mapping = {
        "crack": {
            "crack": "damage",
        },

        "paint": {
            "wear": "wear",
            "paint_damage": "wear",
            "peeling_paint": "wear",
            "discoloration": "wear",
        },

        "mold": {
            "mold": "damage",
            "crack": "damage",
            "wear": "wear",
        },

        "mold2": {
            "mold": "damage",
        },

        "house": {
            "damage": "damage",
            "nodamage": "no_damage",
            "no_damage": "no_damage",
        },

        "surface damage": {
            "damage": "damage",
            "surface_damage": "damage",
        },

        "asbestos": {
            "asbestos": "damage",
        },
    }

    if dataset_name not in mapping:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if raw_label not in mapping[dataset_name]:
        raise ValueError(
            f"Unknown label '{raw_label}' for dataset '{dataset_name}'"
        )

    return mapping[dataset_name][raw_label]