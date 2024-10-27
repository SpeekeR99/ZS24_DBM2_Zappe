def race_to_faction(race):
    ally_races = ["human", "dwarf", "night_elf", "gnome", "draenei", "worgen"]
    horde_races = ["orc", "undead", "tauren", "troll", "blood_elf", "goblin"]

    if race in ally_races:
        return "alliance"
    elif race in horde_races:
        return "horde"
    else:
        return "neutral"


def game_map_to_format(game_map):
    maps_10_v_10 = ["Warsong Gulch", "Twin Peaks", "Battle for Gilneas", "Temple of Kotmogu", "Silvershard Mines"]
    maps_15_v_15 = ["Arathi Basin", "Eye of the Storm", "Deepwind Gorge"]
    maps_40_v_40 = ["Alterac Valley", "Isle of Conquest", "The Battle for Gilneas"]

    if game_map in maps_10_v_10:
        return "10v10"
    elif game_map in maps_15_v_15:
        return "15v15"
    elif game_map in maps_40_v_40:
        return "40v40"
    else:
        return "unknown"
