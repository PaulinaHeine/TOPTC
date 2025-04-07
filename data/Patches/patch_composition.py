import random

PLASTIC_TYPES = {
    "bottle":    {"density": 0.95, "unit_weight": 0.5, "unit_area": 0.2},
    "net":       {"density": 0.92, "unit_weight": 0.7, "unit_area": 1.5},
    "canister":  {"density": 0.94, "unit_weight": 0.6, "unit_area": 0.5},
    "fragments": {"density": 1.02, "unit_weight": 0.2, "unit_area": 0.1},
    "film":      {"density": 0.91, "unit_weight": 0.3, "unit_area": 0.15},
    "foam":      {"density": 0.05, "unit_weight": 0.1, "unit_area": 0.05},
    "microplastic": {"density": 1.05, "unit_weight": 0.01, "unit_area": 0.01}
}

# TODO drift_factor warum definieren wir den der setzt sich dich aus eigenschaftenzusammen?
def calculate_patch_properties(composition):
    total_weight = 0.0
    total_density = 0.0
    total_area = 0.0
    total_count = 0

    for item, count in composition.items():
        props = PLASTIC_TYPES.get(item, {})
        if not props:
            continue
        total_weight += count * props["unit_weight"]
        total_density += count * props["density"]
        total_area += count * props["unit_area"]
        total_count += count

    if total_count == 0:
        return {"density": 1.0, "total_weight": 0.0, "area": 0.0}

    return {
        "patch_density": total_density / total_count,
        "patch_weight": total_weight,
        "patch_area": total_area
    }

def generate_random_patch(min_items=1, max_items=7, min_count=20, max_count=400):
    items = random.sample(list(PLASTIC_TYPES.keys()), k=random.randint(min_items, max_items))
    composition = {item: random.randint(min_count, max_count) for item in items}
    properties = calculate_patch_properties(composition)
    return {"composition": composition, "properties": properties}

patch = generate_random_patch()
print(patch)
