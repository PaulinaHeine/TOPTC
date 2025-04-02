import random

PLASTIC_TYPES = {
    "bottle":    {"drift_factor": 1.0, "density": 0.95, "unit_weight": 0.5, "unit_area": 0.2, "shape": 1.2, "elasticity": 0.3},
    "net":       {"drift_factor": 1.3, "density": 0.92, "unit_weight": 0.7, "unit_area": 1.5, "shape": 2.0, "elasticity": 0.9},
    "canister":  {"drift_factor": 1.2, "density": 0.94, "unit_weight": 0.6, "unit_area": 0.5, "shape": 1.0, "elasticity": 0.2},
    "fragments": {"drift_factor": 0.9, "density": 1.02, "unit_weight": 0.2, "unit_area": 0.1, "shape": 0.7, "elasticity": 0.1},
    "film":      {"drift_factor": 1.1, "density": 0.91, "unit_weight": 0.3, "unit_area": 0.15, "shape": 1.5, "elasticity": 0.8},
    "foam":      {"drift_factor": 1.5, "density": 0.05, "unit_weight": 0.1, "unit_area": 0.05, "shape": 0.5, "elasticity": 0.6},
    "microplastic": {"drift_factor": 0.8, "density": 1.05, "unit_weight": 0.01, "unit_area": 0.01, "shape": 0.3, "elasticity": 0.05},
}

def calculate_patch_properties(composition):
    total_weight = 0.0
    total_drift = 0.0
    total_density = 0.0
    total_area = 0.0
    total_shape = 0.0
    total_elasticity = 0.0
    total_count = 0

    for item, count in composition.items():
        props = PLASTIC_TYPES.get(item, {})
        if not props:
            continue
        total_weight += count * props["unit_weight"]
        total_drift += count * props["drift_factor"]
        total_density += count * props["density"]
        total_area += count * props["unit_area"]
        total_shape += count * props["shape"]
        total_elasticity += count * props["elasticity"]
        total_count += count

    if total_count == 0:
        return {"drift_factor": 1.0, "density": 1.0, "total_weight": 0.0, "area": 0.0, "shape": 1.0, "elasticity": 0.0}

    return {
        "drift_factor": total_drift / total_count,
        "patch_density": total_density / total_count,
        "patch_weight": total_weight,
        "patch_area": total_area,
        "patch_shape": total_shape / total_count,
        "patch_elasticity": total_elasticity / total_count,
    }

def generate_random_patch(min_items=3, max_items=5, min_count=1, max_count=20):
    items = random.sample(list(PLASTIC_TYPES.keys()), k=random.randint(min_items, max_items))
    composition = {item: random.randint(min_count, max_count) for item in items}
    properties = calculate_patch_properties(composition)
    return {"composition": composition, "properties": properties}

patch = generate_random_patch()
print(patch)
