import torch

ROLE_MAP = {"TOP": 0, "JUNGLE": 1, "MID": 2, "ADC": 3, "SUPPORT": 4}
ROLES = ["top", "jng", "mid", "adc", "sup"]
N_ROLES = 5

INDIVIDUAL_STATS = [
    "level", "kills", "deaths", "assists", "cs", "golds", "vision_score",
    "solo_kills", "double_kills", "triple_kills", "quadra_kills", "penta_kills",
    "gd_at_15", "csd_at_15", "xpd_at_15", "objectives_stolen",
    "damage_dealt_to_buildings", "total_heal", "total_heals_on_teammates",
    "damage_self_mitigated", "total_damage_shielded_on_teammates",
    "total_time_cc_dealt", "total_damage_taken", "total_time_spent_dead",
    "shutdown_bounty_collected", "shutdown_bounty_lost", "total_damage_to_champion",
]

STAT_CONFIG = {
    "level": {
        "alpha": 0.02,
        "gamma": torch.tensor([15.9, 14.7, 16.2, 15.3, 11.9]),
        "tau": 1.5,
    },
    "kills": {
        "alpha": 0.05,
        "gamma": torch.tensor([1.6, 1.8, 2.1, 3.4, -0.6]),
        "tau": 2.5,
    },
    "deaths": {
        "alpha": -0.05,
        "gamma": torch.tensor([4.3, 4.3, 3.8, 3.6, 4.9]),
        "tau": 2.0,
    },
    "assists": {
        "alpha": 0.05,
        "gamma": torch.tensor([4.2, 6.3, 5.2, 4.5, 8.8]),
        "tau": 4.0,
    },
    "cs": {
        "alpha": 1.0,
        "gamma": torch.tensor([227.4, 183.0, 256.8, 280.3, 8.0]),
        "tau": 50.0,
    },
    "golds": {
        "alpha": 30.0,
        "gamma": torch.tensor([11628.0, 10947.0, 12453.0, 13569.0, 7224.0]),
        "tau": 2500.0,
    },
    "vision_score": {
        "alpha": 0.3,
        "gamma": torch.tensor([26.4, 44.3, 29.2, 31.1, 107.4]),
        "tau": 15.0,
    },
    "solo_kills": {
        "alpha": 0.01,
        "gamma": torch.tensor([1.05, 0.85, 0.85, 0.95, 0.85]),
        "tau": 0.5,
    },
    "double_kills": {
        "alpha": 0.005,
        "gamma": torch.tensor([0.175, 0.175, 0.175, 0.375, -0.125]),
        "tau": 0.5,
    },
    "triple_kills": {
        "alpha": 0.002,
        "gamma": torch.tensor([-0.05, -0.05, 0.05, 0.15, -0.05]),
        "tau": 0.25,
    },
    "quadra_kills": {
        "alpha": 0.001,
        "gamma": torch.tensor([-0.025, -0.025, -0.025, -0.025, -0.025]),
        "tau": 0.15,
    },
    "penta_kills": {
        "alpha": 0.0005,
        "gamma": torch.tensor([-0.0125, -0.0125, -0.0125, -0.0125, -0.0125]),
        "tau": 0.05,
    },
    "gd_at_15": {
        "alpha": 10.0,
        "gamma": torch.tensor([-250.0, -250.0, -250.0, -250.0, -250.0]),
        "tau": 700.0,
    },
    "csd_at_15": {
        "alpha": 0.2,
        "gamma": torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0]),
        "tau": 15.0,
    },
    "xpd_at_15": {
        "alpha": 10.0,
        "gamma": torch.tensor([-250.0, -250.0, -250.0, -250.0, -250.0]),
        "tau": 650.0,
    },
    "objectives_stolen": {
        "alpha": 0.005,
        "gamma": torch.tensor([-0.125, -0.025, -0.125, -0.125, -0.125]),
        "tau": 0.3,
    },
    "damage_dealt_to_buildings": {
        "alpha": 30.0,
        "gamma": torch.tensor([3367.0, 1099.0, 2934.0, 4005.0, 120.0]),
        "tau": 3000.0,
    },
    "total_heal": {
        "alpha": 20.0,
        "gamma": torch.tensor([5963.0, 21828.0, 4251.0, 4944.0, 4699.0]),
        "tau": 5000.0,
    },
    "total_heals_on_teammates": {
        "alpha": 5.0,
        "gamma": torch.tensor([-107.0, 144.0, -83.0, -12.0, 1935.0]),
        "tau": 2000.0,
    },
    "damage_self_mitigated": {
        "alpha": 100.0,
        "gamma": torch.tensor([29129.0, 32430.0, 14275.0, 8382.0, 15368.0]),
        "tau": 12000.0,
    },
    "total_damage_shielded_on_teammates": {
        "alpha": 5.0,
        "gamma": torch.tensor([-39.0, -45.0, 266.0, -57.0, 1189.0]),
        "tau": 1200.0,
    },
    "total_time_cc_dealt": {
        "alpha": 2.0,
        "gamma": torch.tensor([332.0, 388.0, 281.0, 147.0, 81.0]),
        "tau": 300.0,
    },
    "total_damage_taken": {
        "alpha": 50.0,
        "gamma": torch.tensor([26094.0, 37897.0, 19227.0, 15053.0, 16378.0]),
        "tau": 8000.0,
    },
    "total_time_spent_dead": {
        "alpha": -1.0,
        "gamma": torch.tensor([122.4, 119.0, 108.8, 94.2, 113.2]),
        "tau": 65.0,
    },
    "shutdown_bounty_collected": {
        "alpha": 2.0,
        "gamma": torch.tensor([83.0, 75.0, 96.0, 126.0, -23.0]),
        "tau": 250.0,
    },
    "shutdown_bounty_lost": {
        "alpha": -2.0,
        "gamma": torch.tensor([181.0, 191.0, 201.0, 211.0, 74.0]),
        "tau": 200.0,
    },
    "total_damage_to_champion": {
        "alpha": 100.0,
        "gamma": torch.tensor([16978.0, 11506.0, 19187.0, 21042.0, 4330.0]),
        "tau": 7000.0,
    },
}
