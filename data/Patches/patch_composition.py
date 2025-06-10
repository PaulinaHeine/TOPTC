import random
import numpy as np

PLASTIC_TYPES = {
    # Haushalts- und Verpackungsabfälle
    "bottle":       {"density": 0.95, "unit_weight": 0.5,  "unit_area": 0.1},   # PET-Flaschen
    "canister":     {"density": 0.94, "unit_weight": 3.6,  "unit_area": 0.3},   # Kanister (HDPE)
    "crate":        {"density": 0.95, "unit_weight": 3.0,  "unit_area": 0.4},   # Transportkisten
    "barrel":       {"density": 0.94, "unit_weight": 8.0,  "unit_area": 1.0},   # Plastikfässer
    "plastic_bin":  {"density": 0.96, "unit_weight": 5.0,  "unit_area": 0.8},   # Mülltonnen

    # Fischerei- und Industrieabfälle
    "net":          {"density": 0.92, "unit_weight": 0.7,  "unit_area": 1.2},   # Fischernetze
    "buoy":         {"density": 0.8,  "unit_weight": 6.0,  "unit_area": 1.2},   # Schwimmkörper / Bojen
    "pvc_pipe":     {"density": 1.4,  "unit_weight": 4.0,  "unit_area": 0.6},   # Rohre (PVC)

    # Klein- und Mikroplastik
    "fragments":    {"density": 1.02, "unit_weight": 0.2,  "unit_area": 0.05},  # Plastikbruchstücke
    "film":         {"density": 0.91, "unit_weight": 0.3,  "unit_area": 0.15},  # Verpackungsfolie
    "foam":         {"density": 0.05, "unit_weight": 0.1,  "unit_area": 0.08},  # Styropor
    "microplastic": {"density": 1.05, "unit_weight": 0.01, "unit_area": 0.0001},# Partikel < 5 mm

    # Weitere typische Funde
    "toy":          {"density": 0.9,  "unit_weight": 0.2,  "unit_area": 0.1},   # Kunststoffspielzeug
    "cap":          {"density": 0.92, "unit_weight": 0.05, "unit_area": 0.01},  # Flaschendeckel
    "toothbrush":   {"density": 1.0,  "unit_weight": 0.03, "unit_area": 0.005}, # Zahnbürsten
    "styrobox":     {"density": 0.03, "unit_weight": 0.2,  "unit_area": 0.8},   # Fischbox aus Styropor
    "bag":          {"density": 0.91, "unit_weight": 0.1,  "unit_area": 1.0},   # Plastiksäcke
}




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



def generate_random_patch():
    # Gewichtete Wahrscheinlichkeiten realistischer Fundhäufigkeit
    weighted_types = [
        "fragments", "fragments", "fragments", "fragments",  # häufiger
        "bottle", "bottle", "film", "bag", "bag",
        "net", "net",
        "canister", "crate", "foam",
        "plastic_bin", "barrel", "pvc_pipe",
        "toy", "toy", "cap", "microplastic", "buoy"
    ]

    # Anzahl verschiedener Objekttypen im Patch
    num_types = np.clip(int(np.random.normal(loc=5, scale=2)), 2, 7)

    # Sample realistisch zusammengesetzte Objekte
    items = random.sample(weighted_types, k=num_types)
    items = list(set(items))  # doppelte vermeiden

    # Größe der Patches (mehr große)
    scale = np.random.lognormal(mean=6, sigma=0.6)  # Mittelwert ca. 400 (200–1000)
    total_count = int(np.clip(scale, 100, 1000))

    # Verteilung innerhalb des Patches
    proportions = np.random.dirichlet(np.ones(len(items)))
    composition = {item: int(total_count * p) for item, p in zip(items, proportions)}

    properties = calculate_patch_properties(composition)

    return {"composition": composition, "properties": properties}

def generate_test_patch():
    composition = {
        "bottle": 100,
        "net": 20,
        "foam": 50
    }
    properties = calculate_patch_properties(composition)
    return {"composition": composition, "properties": properties}


import numpy as np
import random

def generate_static_patch( seed=None):
    combined_seed = seed
    random.seed(combined_seed)
    np.random.seed(combined_seed)

    weighted_types = [
        "fragments", "fragments", "fragments", "fragments",
        "bottle", "bottle", "film", "bag", "bag",
        "net", "net",
        "canister", "crate", "foam",
        "plastic_bin", "barrel", "pvc_pipe",
        "toy", "toy", "cap", "microplastic", "buoy"
    ]

    num_types = np.clip(int(np.random.normal(loc=5, scale=2)), 2, 7)
    #items = random.sample(weighted_types, k=num_types)
    items = []
    seen = set()
    while len(items) < num_types:
        item = random.choice(weighted_types)
        if item not in seen:
            seen.add(item)
            items.append(item)

    scale = np.random.lognormal(mean=6, sigma=0.6)
    total_count = int(np.clip(scale, 100, 1000))

    proportions = np.random.dirichlet(np.ones(len(items)))
    composition = {item: int(total_count * p) for item, p in zip(items, proportions)}

    properties = calculate_patch_properties(composition)

    return {"composition": composition, "properties": properties}


