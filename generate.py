import custom
from custom.layers import *
from custom.config import config
from model import MusicTransformer
from utils import find_files_by_extensions
from midi_processor.processor import decode_midi, encode_midi

import datetime

from tensorboardX import SummaryWriter


parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = SummaryWriter(gen_log_dir)

mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=0,
    debug=False)

dir_path = config.load_path

files = list(find_files_by_extensions(dir_path, ['.pickle']))
paths = files[int(len(files) * 0.9):]
len_paths = len(paths)
half_length = int(config.length/2)
paths = [path[:-7].replace("pickle", "midi") for path in paths]

mt.load_state_dict(torch.load(args.model_dir+'_ushiro_nomask/final.pth'))
mt.test()

split_path = config.save_path1.split(".")
split_path2 = config.save_path2.split(".")

for i, path in enumerate(paths):
    midi_data = encode_midi(path)
    data = midi_data[:half_length] + [config.pad_token] * config.length + midi_data[half_length*3:half_length*4]
    inputs = np.array([data])

    inputs = torch.from_numpy(inputs)
    result = mt(inputs, config.length, gen_summary_writer)
    decode_midi(result, file_path=f"{split_path[0]}{i}.mid")

    middle = result[half_length:-half_length]
    decode_midi(middle, file_path=f"{split_path2[0]}{i}.mid")


gen_summary_writer.close()
