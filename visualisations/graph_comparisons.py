from unicodedata import lookup

import matplotlib.pyplot as plt
#from misc.main_greedy import run_greedy  # weil für vgl der masterarbeit erstmal fokus auf greedy
from Modelle.GreedyBoat import run_greedy


# Einstellungen für die Simulation
params = {
    'time_frame': 200,
    'plastic_radius': 10,
    'plastic_number': 100,
    'plastic_seed': 1,
    'boat_number': 2,
    'speed_factor_boat': 3,

    'adaptive_alpha_mode': 0,

    'enable_retargeting': 1,
    'retarget_threshold': 0.999,
     'opportunistic_alpha': 0.9,
}
# Run-Simulation
logbook = []
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]



for i in alphas:
    print("alpha:", i)
    h, *_ = run_greedy(
        weighted_alpha_value=i,
        animation=False,
        **params
    )
    logbook.append(h)



# Nur die dritten Einträge extrahieren
logbook_all = [eintrag[2] for eintrag in logbook]

# Plot-Funktion
def vgl_graph(logbook_all, alphas, params):
    values = [item[0] for item in logbook_all]
    distances = [item[1] for item in logbook_all]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Value', color='tab:blue')
    ax1.plot(alphas, values, marker='o', color='tab:blue', label='Value')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Distance (km)', color='tab:orange')
    ax2.plot(alphas, distances, marker='s', color='tab:orange', label='Distance')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Titel
    plt.title('Value and Distance as a Function of Alpha (Dual-Axis)', fontsize=14)

    # Parameter-Textbox
    param_text = '\n'.join([f'{key}: {value}' for key, value in params.items()])
    plt.gcf().text(0.01, 0.01, f'Simulation Parameters:\n{param_text}', fontsize=9, va='bottom', ha='left',
                   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Aufruf
vgl_graph(logbook_all, alphas, params)

print("Logbook:", logbook)
print("Logbook_all:", logbook_all)
print("Values:", [item[0] for item in logbook_all])
print("Distances:", [item[1] for item in logbook_all])

