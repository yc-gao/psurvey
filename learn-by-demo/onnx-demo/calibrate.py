#!/usr/bin/env python3
import argparse
import os
import glob

import numpy as np
from PIL import Image
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader


class ImageNetDataReader(CalibrationDataReader):
    def __init__(self, images_folder, width, height, batch_size=8):
        self.images_folder = images_folder
        self.width = width
        self.height = height
        self.batch_size = batch_size

        self.label2idx = {
            label: idx for idx, label in enumerate(sorted(entry.name for entry in os.scandir(self.images_folder) if entry.is_dir()))
        }

        self.img_filenames = glob.glob('*/*.JPEG', root_dir=self.images_folder)
        self.img_filenames = np.reshape(
            self.img_filenames[0:int(
                len(self.img_filenames) // self.batch_size * self.batch_size)],
            (-1, self.batch_size))

        self.data_idx = 0

    def __len__(self):
        return len(self.img_filenames)

    def __next__(self):
        if self.data_idx >= len(self):
            raise StopIteration
        batch_filenames = self.img_filenames[self.data_idx]
        self.data_idx = self.data_idx + 1
        return self.load_batch(batch_filenames), [self.label2idx[os.path.dirname(x)] for x in batch_filenames]

    def get_next(self):
        if self.data_idx >= len(self):
            return None
        batch_filenames = self.img_filenames[self.data_idx]
        self.data_idx = self.data_idx + 1

        return {
            'data': self.load_batch(batch_filenames),
            '495': [self.label2idx[os.path.dirname(x)] for x in batch_filenames]
        }

    def rewind(self):
        self.data_idx = 0

    def load_image(self, image_filepath):
        pillow_img = Image.new("RGB", (self.width, self.height))
        pillow_img.paste(
            Image.open(image_filepath)
            .resize((self.width, self.height))
        )
        input_data = np.float32(pillow_img) / 255
        input_data = input_data - \
            np.array([0.485, 0.456, 0.406], dtype=np.float32)
        input_data = input_data / \
            np.array([0.229, 0.224, 0.225], dtype=np.float32)
        nhwc_data = np.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(
            0, 3, 1, 2)  # ONNX Runtime standard
        return nchw_data

    def load_batch(self, batch_filenames):
        unconcatenated_batch_data = []
        for image_name in batch_filenames:
            unconcatenated_batch_data.append(
                self.load_image(self.images_folder + '/' + image_name))
        batch_data = np.concatenate(
            unconcatenated_batch_data, axis=0
        )
        return batch_data


class ImageNetPipeline:
    @staticmethod
    def eval(sess, data_reader):
        total_count = 0
        correct_count = 0
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name
        while True:
            batch = data_reader.get_next()
            if not batch:
                break
            pred, = sess.run(None, {f"{iname}": batch[iname]})
            pred = np.argmax(pred, axis=1)

            total_count = total_count + len(batch[oname])
            correct_count = correct_count + (pred == batch[oname]).sum()
        return correct_count / total_count

    @staticmethod
    def pred(sess, X):
        iname = sess.get_inputs()[0].name
        pred, = sess.run(None, {f"{iname}": X})
        pred = np.argmax(pred, axis=1)
        return pred


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()
    data_reader = ImageNetDataReader(options.dataset_path, 224, 224)
    sess = ort.InferenceSession(options.model, providers=[
                                'CUDAExecutionProvider'])

    acc = ImageNetPipeline.eval(sess, data_reader)
    print(f"acc: {acc * 100:.4f}%")


if __name__ == "__main__":
    main()
