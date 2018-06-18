import { tensor2d, util } from '@tensorflow/tfjs'
import axios from 'axios'

import { Dataset } from '../types/models'

const CATS_DATASET = '/cats1000.bin'
const RAINBOWS_DATASET = '/rainbows1000.bin'
const TRAINS_DATASET = '/trains1000.bin'
const IMAGE_SIZE = 784

export class GoogleDoodlesDataset implements Dataset {
  private shuffledTrainIndex: number
  private shuffledTestIndex: number
  private trainIndices: any
  private testIndices: any
  private trainImages: any
  private trainLabels: any
  private testImages: any
  private testLabels: any
  private imageCount: number
  private classCount: number

  public constructor () {
    this.shuffledTrainIndex = 0
    this.shuffledTestIndex = 0
  }

  public load = async () => {
    const config = {
      responseType: 'arraybuffer'
    }
    const res = await Promise.all([
      axios.get(CATS_DATASET, config),
      axios.get(RAINBOWS_DATASET, config),
      axios.get(TRAINS_DATASET, config)
    ])

    const sets = res.map((r) => new Uint8Array(r.data))
    this.imageCount = sets.reduce((p, c) => p + c.length, 0) / IMAGE_SIZE
    this.classCount = sets.length

    const data = new Uint8Array(this.imageCount * IMAGE_SIZE)
    const labels = new Uint8Array(this.imageCount * this.classCount)

    let loadingOffset = 0
    sets.forEach((d, i) => {
      data.set(d, loadingOffset * IMAGE_SIZE)

      const images = d.length / IMAGE_SIZE

      const label = new Uint8Array(this.classCount)
      label[i] = 1

      for (let i = 0; i < images; i++) {
        labels.set(label, (i + loadingOffset) * this.classCount)
      }

      loadingOffset += images
    })

    const trainCount = Number.parseInt(`${this.imageCount * 0.8}`, 10)
    const testCount = this.imageCount - trainCount

    this.trainIndices = util.createShuffledIndices(trainCount)
    this.testIndices = util.createShuffledIndices(testCount)

    // Slice the the images and labels into train and test sets.
    this.trainImages = data.slice(0, IMAGE_SIZE * trainCount)
    this.trainLabels = labels.slice(0, this.classCount * trainCount)
    this.testImages = data.slice(IMAGE_SIZE * trainCount)
    this.testLabels = labels.slice(this.classCount * trainCount)
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
    const batchLabelsArray = new Uint8Array(batchSize * this.classCount)

    for (let i = 0; i < batchSize; i++) {
      const idx = index()

      const image =
        data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE)

      batchImagesArray.set(image, i * IMAGE_SIZE)

      const label = data[1].slice(idx * this.classCount, idx * this.classCount + this.classCount)
      batchLabelsArray.set(label, i * this.classCount)
    }

    // Reshape the training data from [64, 28x28] to [64, 28, 28, 1] so
    // that we can feed it to our convolutional neural net.
    const input = tensor2d(batchImagesArray, [ batchSize, IMAGE_SIZE ]).reshape([ batchSize, 28, 28, 1 ])
    const expected = tensor2d(batchLabelsArray, [ batchSize, this.classCount ])

    return {
      input,
      expected
    }
  }
}
