import numpy as np
import matplotlib.pyplot as plt
from .base_class import HierarchicalDatasetBase
from .const import LANGUAGE_DATASET_PARAMS, AUTOREG_PARAMS, SHIFT_HIGH_T, SHIFT_LOW_T, SHIFT_HIGH_V, SHIFT_LOW_V


class FormDataset(HierarchicalDatasetBase):

    def __init__(self, analyses, shift_high=0, shift_low=0, max_l=256, h=12, n_channels=8,
                 multi_phrase_label=False, random_pitch_aug=True):
        super(FormDataset, self).__init__(
            analyses, shift_high, shift_low, max_l, h, n_channels,
            use_autoreg_cond=False, use_external_cond=False,
            multi_phrase_label=multi_phrase_label, random_pitch_aug=random_pitch_aug, mask_background=False)

        assert max_l >= 210, "Some pieces may be longer than the current max_l."

        form_langs = [analysis['languages']['form'] for analysis in analyses]

        self.key_rolls = [form_lang['key_roll'] for form_lang in form_langs]
        self.phrase_rolls = [form_lang['phrase_roll'][:, :, np.newaxis] for form_lang in form_langs]

        self.lengths = np.array([roll.shape[1] for roll in self.key_rolls])

        self.start_ids_per_song = [np.zeros(1, dtype=np.int64) for _ in range(len(self.lengths))]

        self.indices = self._song_id_to_indices()

        # Song embeddings taken from analyses (training data) (train_analyses from data_utils.__init__.py)
        try:
            self.embeddings = [analysis.get('embedding', None) for analysis in analyses]
        except Exception:
            self.embeddings = None

    def get_data_sample(self, song_id, start_id, shift):
        self.store_key(song_id, shift)
        self.store_phrase(song_id)

        # Converts stored data to image tensor format
        img = self.lang_to_img(song_id, start_id, end_id=start_id + self.max_l, tgt_lgth=self.max_l)

        ####
        # Always attach song embedding as fourth element if available
        # song_emb = self.get_song_embedding(song_id)

        return img, None, None

        # return img, None, None, song_emb

    def lang_to_img(self, song_id, start_id, end_id, tgt_lgth=None):
        '''
        song_id: ID of the song to extract the data from
        start_id: Starting time step index for the segment to extract
        end_id: Ending time step index for the segment to extract
        tgt_lgth: Length of the target image to extract. If None, defaults to end_id - start_id

        Returns: ??


        '''
        # Extracts key information for the specified time segment
        # Shape: (2 channels, length L, 12 pitch classes)
        key_roll = self._key[:, start_id: end_id]  # (2, L, 12)

        # Extracts phrase structure information for the specified time segment
        # Shape: (6 channels, length L, 1 dimension)
        phrase_roll = self._phrase[:, start_id: end_id]  # (6, L, 1)

        actual_l = self._key.shape[1]

        # to output image
        if tgt_lgth is None:
            tgt_lgth = end_id - start_id

        # Creates a zero-initialized image tensor with shape (8, tgt_lgth, 16) for form dataset
        # Note: Check out LANGUAGE_DATASET_PARAMS in const.py. n_channels = 8, h = 16
        img = np.zeros((self.n_channels, tgt_lgth, self.h), dtype=np.float32)

        # Channels 0-1: Key roll data (2 channels x 12 pitch classes, length L)
        img[0: 2, 0: actual_l, 0: 12] = self._key

        # Channels 2-7: Phrase roll data (6 channels x 1 dimension, length L)
        img[2: 8, 0: actual_l] = self._phrase


        # 💡 DEBUG PRINTS
        # print(f"\n--- DEBUG: lang_to_img() (FORM) ---")
        # print(f"-------------------------\n")
        # print(f"song_id: {song_id}, start_id: {start_id}, end_id: {end_id}")
        # print(f"key_roll shape: {key_roll.shape}, phrase_roll shape: {phrase_roll.shape}")
        # print(f"img shape: {img.shape}")
        # print(f"\nkey_roll sample:\n{key_roll}")
        # print(f"\nphrase_roll sample:\n{phrase_roll}")
        # # print(f"\nimg sample (first channel slice):\n{img[0]}")
        # print(f"\n")

        # print("Summary of img's 8 channels:")
        '''
        FIRST 2 CHANNELS: Key information
        Channel 0: What is the key (tonic) for every measure?
        Channel 1: Defining notes in the scale (1 means raised)

        LAST 6 CHANNELS: Phrase information (see Table 3 in paper)
        Channel 2: "A" Verse section phrases
        Channel 3: "B" 1 Chorus section phrases
        Channel 4: "X" 2 Other phrases with lead melody
        Channel 5: "i" 3 Intro section phrases
        Channel 6: "o" 4 Outro section phrases
        Channel 7: "b" 5 Bridge section phrases
        I think 1 channel has values at a time. if 1 channel is activated, the others have 0s.
        #s denote that that phrase is currently active in that timestep. is a countdown to the end of that phrase.
        has 12 values because the phrase was broadcasted to match the dimensions of the key information.

        img tensor: width is 256 measures always (defined this way for the form level)
        '''
        # print summary of all 8 channels
        # for c in range(img.shape[0]):
        #     print(f"Channel {c}: ")
        #     print(f"img[{c}, :5, :12] =\n{img[c, :5, :12]}\n")
            
        #     print(f"-------------------------\n")

        return img

    def show(self, item, show_img=True):
        sample = self[item][0]
        titles = ['key', 'phrase0-1', 'phrase2-3', 'phrase4-5']

        if show_img:
            fig, axs = plt.subplots(4, 1, figsize=(10, 30))
            for i in range(4):
                img = sample[2 * i: 2 * i + 2]
                img = np.pad(img, pad_width=((0, 1), (0, 0), (0, 0)), mode='constant')
                img = img.transpose((2, 1, 0))
                axs[i].imshow(img, origin='lower', aspect='auto')
                axs[i].title.set_text(titles[i])
            plt.show()


def create_form_datasets(train_analyses, valid_analyses, multi_phrase_label=False, random_pitch_aug=True):

    lang_params = LANGUAGE_DATASET_PARAMS['form']

    train_dataset = FormDataset(
        train_analyses, SHIFT_HIGH_T, SHIFT_LOW_T, lang_params['max_l'], lang_params['h'], lang_params['n_channel'],
        multi_phrase_label=multi_phrase_label, random_pitch_aug=random_pitch_aug
    )
    valid_dataset = FormDataset(
        valid_analyses, SHIFT_HIGH_V, SHIFT_LOW_V, lang_params['max_l'], lang_params['h'], lang_params['n_channel'],
        multi_phrase_label=multi_phrase_label, random_pitch_aug=random_pitch_aug
    )
    return train_dataset, valid_dataset
