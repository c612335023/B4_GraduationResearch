from midi2audio import FluidSynth
from progress.bar import Bar

fs = FluidSynth(sound_font='font.sf2')

for i in Bar('midi2wav').iter(range(1)):
    fs.midi_to_audio('bin_proposed/generated'+str(i)+'.mid', 'dataset/wav_proposed/proposed'+str(i)+'.wav')