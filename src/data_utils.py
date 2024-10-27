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


class Player:
    def __init__(self, player_id, race, cls, winning_team, kills, deaths, damage_done, healing_done, damage_taken, healing_taken):
        self.player_id = player_id
        self.race = race
        self.cls = cls
        self.winning_team = winning_team
        self.kills = kills
        self.deaths = deaths
        self.damage_done = damage_done
        self.healing_done = healing_done
        self.damage_taken = damage_taken
        self.healing_taken = healing_taken
        self.faction = race_to_faction(race)


class BattleGround:
    def __init__(self, match_id, game_map, start_time, duration):
        self.match_id = match_id
        self.game_map = game_map
        self.start_time = start_time
        self.duration = duration
        self.players = []
        self.format = game_map_to_format(game_map)
