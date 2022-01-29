import json
import pickle
import os
import threading


class IMDB:

    def __init__(self, imdb_path, frames_size=5000, dump_index=0, size=0,
                 save_method=None, load_method=None):
        self.imdb_path = imdb_path
        self._frames = []
        self._frames_size = frames_size
        self._dump_index = dump_index
        self._size = size
        self._lock = threading.Lock()
        self._save_method = save_method
        self._load_method = load_method

    @classmethod
    def load_from_disk(cls, imdb_path, load_method=None):
        dataset = json.load(open(os.path.join(imdb_path, 'dataset.json')))
        return cls(imdb_path,
                   size=int(dataset['size']),
                   dump_index=-1,
                   frames_size=int(dataset['frames_size']),
                   load_method=load_method)

    def _are_frames_full(self):
        if len(self._frames) == self._frames_size:
            return True
        return False

    def _get_filepath(self, start_index):
        return os.path.join(self.imdb_path, 'samples_{0:06d}.pkl'.format(start_index))

    def _dump_frames(self):
        start_index = self._dump_index * self._frames_size
        self._dump_index += 1
        filepath = self._get_filepath(start_index)
        if self._save_method is None:
            with open(filepath, 'wb') as fd:
                pickle.dump(self._frames, fd)
        else:
            self._save_method(self._frames, filepath)
        self._frames = []

    def append(self, frame):
        if self._are_frames_full():
            self._dump_frames()
        self._frames.append(frame)

    def save(self):
        self._size = self._dump_index * self._frames_size + len(self._frames)
        self._dump_frames()
        dataset = {
            'path': self.imdb_path,
            'size': self._size,
            'frames_size': self._frames_size
        }
        json.dump(dataset,
                  open(os.path.join(self.imdb_path, 'dataset.json'), 'w'),
                  indent=4)

    def _load_frames(self, index):
        asked_file_index = int(index / self._frames_size)
        if not asked_file_index == self._dump_index:
            filepath = self._get_filepath(asked_file_index * self._frames_size)
            if self._load_method is None:
                self._frames = pickle.load(open(filepath, 'rb'))
            else:
                self._frames = self._load_method(filepath)
            self._dump_index = asked_file_index
        return index - (self._frames_size * asked_file_index)

    def __getitem__(self, index):
        with self._lock:
            active_index = self._load_frames(index)
            return self._frames[active_index]

    def __len__(self):
        return self._size

    def __del__(self):
        del self.imdb_path
        del self._frames
        del self._frames_size
        del self._dump_index
        del self._size
        del self._lock
