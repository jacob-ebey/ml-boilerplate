/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import { tensor2d, util } from '@tensorflow/tfjs'
import { Dataset } from '../types/models'

const IMAGE_SIZE = 784
const NUM_CLASSES = 10
const NUM_DATASET_ELEMENTS = 65000

const NUM_TRAIN_ELEMENTS = 55000
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS

const MNIST_IMAGES_SPRITE_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png'
const MNIST_LABELS_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8'

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistDataset implements Dataset {
  private shuffledTrainIndex: number
  private shuffledTestIndex: number
  private datasetImages: any
  private datasetLabels: any
  private trainIndices: any
  private testIndices: any
  private trainImages: any
  private testImages: any
  private trainLabels: any
  private testLabels: any

  constructor () {
    this.shuffledTrainIndex = 0
    this.shuffledTestIndex = 0
  }

  async load () {
    // Make a request for the MNIST sprited image.
    const img = new Image()
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      throw new Error('Could not create a 2D context on a canvas.')
    }

    const imgRequest = new Promise<Float32Array>((resolve, reject) => {
      img.crossOrigin = ''
      img.onload = () => {
        img.width = img.naturalWidth
        img.height = img.naturalHeight

        const datasetBytesBuffer =
          new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4)

        const chunkSize = 5000
        canvas.width = img.width
        canvas.height = chunkSize

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize)
          ctx.drawImage(
            img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
            chunkSize)

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255
          }
        }

        this.datasetImages = new Float32Array(datasetBytesBuffer)

        resolve(this.datasetImages)
      }

      img.src = MNIST_IMAGES_SPRITE_PATH
    })

    const labelsRequest = fetch(MNIST_LABELS_PATH)
    // tslint: disable-next-line
    const [ datasetImages, labelsResponse ] =
    await Promise.all([ imgRequest, labelsRequest ])

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer())

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = util.createShuffledIndices(NUM_TRAIN_ELEMENTS)
    this.testIndices = util.createShuffledIndices(NUM_TEST_ELEMENTS)

    // Slice the the images and labels into train and test sets.
    this.trainImages =
      datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS)
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS)
    this.trainLabels =
      this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS)
    this.testLabels =
      this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS)
  }

  public nextTrainBatch (batchSize: number) {
    return this.nextBatch(
      batchSize, [ this.trainImages, this.trainLabels ], () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length
        return this.trainIndices[this.shuffledTrainIndex]
      })
  }

  public nextTestBatch (batchSize: number) {
    return this.nextBatch(batchSize, [ this.testImages, this.testLabels ], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length
      return this.testIndices[this.shuffledTestIndex]
    })
  }

  public nextBatch (batchSize: number, data: any, index: () => number) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE)
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES)

    for (let i = 0; i < batchSize; i++) {
      const idx = index()

      const image =
        data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE)
      batchImagesArray.set(image, i * IMAGE_SIZE)

      const label =
        data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES)
      batchLabelsArray.set(label, i * NUM_CLASSES)
    }

    // Reshape the training data from [64, 28x28] to [64, 28, 28, 1] so
    // that we can feed it to our convolutional neural net.
    const input = tensor2d(batchImagesArray, [ batchSize, IMAGE_SIZE ]).reshape([ batchSize, 28, 28, 1 ])
    const expected = tensor2d(batchLabelsArray, [ batchSize, NUM_CLASSES ])

    return {
      input,
      expected
    }
  }
}
