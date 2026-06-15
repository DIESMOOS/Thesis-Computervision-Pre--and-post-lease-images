def map_source_label(dataset_name: str, raw_label: str) -> str:
    dataset_name = dataset_name.lower().strip()
    raw_label = raw_label.lower().strip()

    mapping = {
        "crack": {
            "crack": "crack",
        },

        "paint": {
            "wear": "wear",
            "paint_damage": "wear",
            "peeling_paint": "wear",
            "discoloration": "wear",
            "metal": "wear",
        },

        "mold": {
            "mold": "mold",
            "mould": "mold",
            "crack": "crack",
            "wear": "wear",
        },

        "mold2": {
            "mold": "mold",
            "mould": "mold",
        },

        "house": {
            "damage": "damage",
            "amber": "damage",
            "red": "damage",
            "green": "no_damage",
            "nodamage": "no_damage",
            "no_damage": "no_damage",
        },

        "surface damage": {
            "damage": "damage",
            "surface_damage": "damage",
            "crack": "crack",
            "erosion": "wear",
        },

        "asbestos": {
            "asbestos": "asbestos",
            "thick-dark-mark": "asbestos",
            "thick-light-mark": "asbestos",
            "thin-dark-mark": "asbestos",
            "thin-light-mark": "asbestos",
        },
    }

    if dataset_name not in mapping:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if raw_label not in mapping[dataset_name]:
        raise ValueError(
            f"Unknown label '{raw_label}' for dataset '{dataset_name}'"
        )

    return mapping[dataset_name][raw_label]