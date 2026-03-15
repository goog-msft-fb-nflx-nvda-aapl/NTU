import numpy as np
import miditoolkit

DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]
DEFAULT_RESOLUTION = 480

class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name; self.start = start; self.end = end
        self.velocity = velocity; self.pitch = pitch
    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(Item('Note', note.start, note.end, note.velocity, note.pitch))
    note_items.sort(key=lambda x: x.start)
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item('Tempo', tempo.time, None, None, int(tempo.tempo)))
    tempo_items.sort(key=lambda x: x.start)
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item('Tempo', tick, None, None, existing_ticks[tick]))
        else:
            output.append(Item('Tempo', tick, None, None, output[-1].pitch))
    return note_items, output

def quantize_items(items, ticks=120):
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items

def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = [item for item in items if db1 <= item.start < db2]
        groups.append([db1] + insiders + [db2])
    return groups

class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name; self.time = time; self.value = value; self.text = text
    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event('Bar', None, None, '{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags - item.start))
            events.append(Event('Position', item.start, '{}/{}'.format(index+1, DEFAULT_FRACTION), str(item.start)))
            if item.name == 'Note':
                vel_idx = np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side='right') - 1
                events.append(Event('Note Velocity', item.start, vel_idx,
                    '{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[vel_idx])))
                events.append(Event('Note On', item.start, item.pitch, str(item.pitch)))
                dur = item.end - item.start
                dur_idx = np.argmin(abs(DEFAULT_DURATION_BINS - dur))
                events.append(Event('Note Duration', item.start, dur_idx,
                    '{}/{}'.format(dur, DEFAULT_DURATION_BINS[dur_idx])))
            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    ts, tv = Event('Tempo Class', item.start, 'slow', None), Event('Tempo Value', item.start, tempo-DEFAULT_TEMPO_INTERVALS[0].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    ts, tv = Event('Tempo Class', item.start, 'mid', None), Event('Tempo Value', item.start, tempo-DEFAULT_TEMPO_INTERVALS[1].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    ts, tv = Event('Tempo Class', item.start, 'fast', None), Event('Tempo Value', item.start, tempo-DEFAULT_TEMPO_INTERVALS[2].start, None)
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    ts, tv = Event('Tempo Class', item.start, 'slow', None), Event('Tempo Value', item.start, 0, None)
                else:
                    ts, tv = Event('Tempo Class', item.start, 'fast', None), Event('Tempo Value', item.start, 59, None)
                events.append(ts); events.append(tv)
    return events

def word_to_event(words, word2event):
    events = []
    for word in words:
        parts = word2event.get(word).split('_', 1)
        events.append(Event(parts[0], None, parts[1], None))
    return events

def write_midi(words, word2event, output_path, prompt_path=None):
    events = word_to_event(words, word2event)
    ticks_per_bar = DEFAULT_RESOLUTION * 4

    # collect notes: look for Position -> Note On, then find nearest Note Velocity + Note Duration
    temp_notes = []
    temp_tempos = []
    current_bar = 0

    i = 0
    current_position = None
    current_tempo_class = None

    while i < len(events):
        ev = events[i]
        if ev.name == 'Bar':
            if i > 0:
                temp_notes.append('Bar')
                temp_tempos.append('Bar')
            current_bar_marker = True
        elif ev.name == 'Position':
            current_position = int(ev.value.split('/')[0]) - 1
        elif ev.name == 'Note On':
            pitch = int(ev.value)
            # look ahead for velocity and duration
            velocity = int(DEFAULT_VELOCITY_BINS[15])  # default mid velocity
            duration = DEFAULT_DURATION_BINS[7]         # default duration
            for j in range(max(0, i-4), i):
                if events[j].name == 'Note Velocity':
                    velocity = int(DEFAULT_VELOCITY_BINS[int(events[j].value)])
            for j in range(i+1, min(len(events), i+5)):
                if events[j].name == 'Note Velocity':
                    velocity = int(DEFAULT_VELOCITY_BINS[int(events[j].value)])
                if events[j].name == 'Note Duration':
                    duration = DEFAULT_DURATION_BINS[int(events[j].value)]
                    break
            if current_position is not None:
                temp_notes.append([current_position, velocity, pitch, duration])
        elif ev.name == 'Note Velocity':
            pass  # handled above
        elif ev.name == 'Note Duration':
            pass  # handled above
        elif ev.name == 'Tempo Class':
            current_tempo_class = ev.value
        elif ev.name == 'Tempo Value':
            if current_tempo_class and current_position is not None:
                if current_tempo_class == 'slow':
                    tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(ev.value)
                elif current_tempo_class == 'mid':
                    tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(ev.value)
                else:
                    tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(ev.value)
                temp_tempos.append([current_position, tempo])
        i += 1

    # convert to absolute time
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            bar_st = current_bar * ticks_per_bar
            bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            notes.append(miditoolkit.Note(velocity, pitch, int(st), int(st + duration)))

    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            bar_st = current_bar * ticks_per_bar
            bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])

    if not tempos:
        tempos = [[0, 120]]

    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        last_time = DEFAULT_RESOLUTION * 4 * 8
        for note in notes:
            note.start += last_time; note.end += last_time
        midi.instruments[0].notes.extend(notes)
        temp_tempos2 = [t for t in midi.tempo_changes if t.time < last_time]
        for st, bpm in tempos:
            temp_tempos2.append(miditoolkit.midi.containers.TempoChange(bpm, st + last_time))
        midi.tempo_changes = temp_tempos2
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        midi.tempo_changes = [miditoolkit.midi.containers.TempoChange(bpm, st) for st, bpm in tempos]

    midi.dump(output_path)
    print(f'  Notes written: {len(notes)}')
